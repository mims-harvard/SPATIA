# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
import random
import time
from typing import Iterable, Optional, Dict
from training.dataloader import CellDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


def sliced_wasserstein_distance(
    feat1: torch.Tensor, 
    feat2: torch.Tensor, 
    num_projections: int = 64
) -> torch.Tensor:
    if feat1.dim() > 2:
        feat1 = feat1.view(feat1.size(0), -1)
    if feat2.dim() > 2:
        feat2 = feat2.view(feat2.size(0), -1)
    
    if feat1.shape != feat2.shape:
        min_d = min(feat1.size(1), feat2.size(1))
        feat1 = feat1[:, :min_d]
        feat2 = feat2[:, :min_d]
    
    dim = feat1.shape[1]
    device = feat1.device
    
    projections = torch.randn((dim, num_projections), device=device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    proj1 = feat1 @ projections
    proj2 = feat2 @ projections
    
    proj1_sorted, _ = torch.sort(proj1, dim=0)
    proj2_sorted, _ = torch.sort(proj2, dim=0)
    
    swd = torch.abs(proj1_sorted - proj2_sorted).mean()
    
    return swd


def compute_weighted_fm_loss(
    model_out: torch.Tensor,
    target_velocity: torch.Tensor,
    confidence: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    raw_loss = torch.mean((model_out - target_velocity) ** 2, dim=[1, 2, 3])
    
    weights = confidence ** alpha
    weights = weights / (weights.mean() + 1e-8)
    
    weighted_loss = (weights * raw_loss).mean()
    
    return weighted_loss


def compute_contrastive_condition_loss(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    target_velocity: torch.Tensor,
    cond_positive: Dict,
    cond_negative: Dict,
    margin: float = 0.1,
) -> torch.Tensor:
    with torch.cuda.amp.autocast():
        v_pred_pos = model(x_t, t, extra=cond_positive)
        loss_pos = torch.sum((v_pred_pos - target_velocity) ** 2, dim=[1, 2, 3])
        
        v_pred_neg = model(x_t, t, extra=cond_negative)
        loss_neg = torch.sum((v_pred_neg - target_velocity) ** 2, dim=[1, 2, 3])
    
    contrastive_loss = torch.relu(loss_pos - loss_neg + margin).mean()
    
    return contrastive_loss


class FeatureExtractor(nn.Module):
    
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(128 * 4 * 4, output_dim)
        
        self._init_weights()
        for param in self.parameters():
            param.requires_grad = False
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def my_train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
    datamodule: CellDataLoader,
    use_initial: int,
    spatia_conditioner: Optional[nn.Module] = None,
    feature_extractor: Optional[nn.Module] = None,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)
    
    epoch_loss_fm = MeanMetric().to(device, non_blocking=True)
    epoch_loss_contrast = MeanMetric().to(device, non_blocking=True)
    epoch_loss_morph = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
    else:
        path = CondOTProbPath()
    
    use_weighted_fm = getattr(args, 'use_weighted_fm', False)
    ot_alpha = getattr(args, 'ot_alpha', 1.0)
    lambda_contrast = getattr(args, 'lambda_contrast', 0.0)
    lambda_morph = getattr(args, 'lambda_morph', 0.0)

    for data_iter_step, batch in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break
        
        x_real, y_trg, y_mod = batch['X'], batch['mols'], batch['y_id']
        x_real_ctrl, x_real_trt = x_real
        x_real_ctrl, x_real_trt = x_real_ctrl.to(device), x_real_trt.to(device)
        y_trg = y_trg.long().to(device)            
        y_org = None 
        
        if 'concat_conditioning' in batch:
            z_emb_trg = batch['concat_conditioning'].to(device)
        else:
            z_emb_trg = datamodule.embedding_matrix(y_trg).to(device)
        
        ot_confidence = batch.get('ot_confidence', None)
        if ot_confidence is not None:
            ot_confidence = ot_confidence.float().to(device)
        
        delta_g = batch.get('delta_g', None)
        if delta_g is not None:
            delta_g = delta_g.float().to(device)
        
        delta_m = batch.get('delta_m', None)
        if delta_m is not None:
            delta_m = delta_m.float().to(device)
        
        transition_ids = batch.get('transition_ids', None)
        if transition_ids is not None:
            transition_ids = transition_ids.long().to(device)
        
        if spatia_conditioner is not None and delta_g is not None and delta_m is not None and transition_ids is not None:
            z_emb_trg = spatia_conditioner(z_emb_trg, transition_ids, delta_g, delta_m)
        
        samples = None
        labels = None
        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"concat_conditioning": z_emb_trg}
        
        if args.discrete_flow_matching:
            samples = (x_real_trt * 0.5 + 0.5)
            samples = (samples * 255.0).to(torch.long)
            t = torch.rand(samples.shape[0]).to(device)
            x_0 = (
                torch.zeros(samples.shape, dtype=torch.long, device=device) + MASK_TOKEN
            )
            path_sample = path.sample(t=t, x_0=x_0, x_1=samples)

            logits = model(path_sample.x_t, t=t, extra=conditioning)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape([-1, 257]), samples.reshape([-1])
            ).mean()
            loss_fm = loss
            loss_contrast = torch.tensor(0.0, device=device)
            loss_morph = torch.tensor(0.0, device=device)
        else:
            if args.skewed_timesteps:
                t = skewed_timestep_sample(x_real_ctrl.shape[0], device=device)
            else:
                t = torch.rand(x_real_ctrl.shape[0]).to(device)
            if use_initial == 1:
                x_0 = x_real_ctrl
            elif use_initial == 2:
                p_r = random.random()
                if p_r > args.noise_prob:
                    x_0 = x_real_ctrl
                else:
                    x_0 = x_real_ctrl + torch.randn(x_real_ctrl.shape, dtype=torch.float32, device=device) * args.noise_level
            else:
                x_0 = torch.randn(x_real_ctrl.shape, dtype=torch.float32, device=device)
            
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_real_trt)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            with torch.cuda.amp.autocast():
                v_pred = model(x_t, t, extra=conditioning)
                
                if use_weighted_fm and ot_confidence is not None:
                    loss_fm = compute_weighted_fm_loss(v_pred, u_t, ot_confidence, alpha=ot_alpha)
                else:
                    loss_fm = torch.pow(v_pred - u_t, 2).mean()
                
                loss_contrast = torch.tensor(0.0, device=device)
                if lambda_contrast > 0 and len(conditioning) > 0:
                    batch_size = x_real_ctrl.size(0)
                    idx_rand = torch.randperm(batch_size, device=device)
                    
                    cond_neg = {}
                    if 'concat_conditioning' in conditioning:
                        cond_neg['concat_conditioning'] = conditioning['concat_conditioning'][idx_rand]
                    
                    loss_contrast = compute_contrastive_condition_loss(
                        model, x_t, t, u_t, conditioning, cond_neg, margin=0.1
                    )
                
                loss_morph = torch.tensor(0.0, device=device)
                if lambda_morph > 0 and feature_extractor is not None:
                    t_expand = t[:, None, None, None]
                    x_1_pred = x_t + (1 - t_expand) * v_pred
                    
                    feat_pred = feature_extractor(x_1_pred)
                    feat_real = feature_extractor(x_real_trt)
                    loss_morph = sliced_wasserstein_distance(feat_pred, feat_real)
                
                loss = loss_fm + lambda_contrast * loss_contrast + lambda_morph * loss_morph

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)
        epoch_loss_fm.update(loss_fm)
        epoch_loss_contrast.update(loss_contrast)
        epoch_loss_morph.update(loss_morph)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            log_msg = f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute():.4f}"
            if use_weighted_fm or lambda_contrast > 0 or lambda_morph > 0:
                log_msg += f" (fm={loss_fm.item():.4f}"
                if lambda_contrast > 0:
                    log_msg += f", contr={loss_contrast.item():.4f}"
                if lambda_morph > 0:
                    log_msg += f", morph={loss_morph.item():.4f}"
                log_msg += ")"
            log_msg += f", lr = {lr}"
            logger.info(log_msg)

    lr_schedule.step()
    
    result = {"loss": float(epoch_loss.compute().detach().cpu())}
    result["loss_fm"] = float(epoch_loss_fm.compute().detach().cpu())
    if lambda_contrast > 0:
        result["loss_contrast"] = float(epoch_loss_contrast.compute().detach().cpu())
    if lambda_morph > 0:
        result["loss_morph"] = float(epoch_loss_morph.compute().detach().cpu())
    
    return result
