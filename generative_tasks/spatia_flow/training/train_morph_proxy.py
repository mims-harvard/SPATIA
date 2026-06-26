
import argparse
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from skimage import filters, measure

logger = logging.getLogger(__name__)


def extract_simple_morphology(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.dtype != np.uint8:
        image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    try:
        thresh_val = filters.threshold_otsu(gray)
        mask = gray > thresh_val
    except ValueError:
        return np.zeros(10, dtype=np.float32)

    labels = measure.label(mask)
    if labels.max() == 0:
        return np.zeros(10, dtype=np.float32)

    regions = measure.regionprops(labels, intensity_image=gray)
    if not regions:
        return np.zeros(10, dtype=np.float32)

    main_region = max(regions, key=lambda r: r.area)

    try:
        features = np.array([
            main_region.area,
            main_region.perimeter if main_region.perimeter else 0.0,
            main_region.eccentricity if hasattr(main_region, 'eccentricity') else 0.0,
            main_region.solidity if hasattr(main_region, 'solidity') else 0.0,
            main_region.mean_intensity,
            main_region.max_intensity,
            main_region.min_intensity,
            np.std(gray[mask]) if np.any(mask) else 0.0,
            main_region.major_axis_length if hasattr(main_region, 'major_axis_length') else 0.0,
            main_region.minor_axis_length if hasattr(main_region, 'minor_axis_length') else 0.0,
        ], dtype=np.float32)
    except Exception:
        return np.zeros(10, dtype=np.float32)

    return features


class MorphProxyEncoder(nn.Module):

    def __init__(self, output_dim: int = 10):
        super().__init__()
        self.output_dim = output_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


class LMDBMorphDataset(data.Dataset):

    def __init__(
        self,
        lmdb_path: str,
        cache_path: Optional[str] = None,
        image_size: int = 256,
        max_samples: Optional[int] = None,
    ):
        self.lmdb_path = lmdb_path
        self.image_size = image_size
        self.env = None

        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
        with env.begin(write=False) as txn:
            self.keys = [
                key.decode("utf-8")
                for key, _ in txn.cursor()
                if not key.decode("utf-8").startswith("__")
            ]
        env.close()

        if max_samples is not None and max_samples < len(self.keys):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(self.keys), max_samples, replace=False)
            self.keys = [self.keys[i] for i in sorted(indices)]

        self.labels = None
        if cache_path and os.path.exists(cache_path):
            cached = np.load(cache_path)
            if len(cached["labels"]) == len(self.keys):
                self.labels = cached["labels"]
                logger.info(f"Loaded cached labels from {cache_path}")

    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, subdir=False
            )
        return self.env

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        env = self._get_env()
        key = self.keys[idx]

        with env.begin(write=False) as txn:
            buf = txn.get(key.encode("utf-8"))

        if buf is None:
            return (
                torch.zeros(3, self.image_size, self.image_size),
                torch.zeros(10),
            )

        try:
            img = Image.open(io.BytesIO(buf)).convert("RGB")
        except Exception:
            return (
                torch.zeros(3, self.image_size, self.image_size),
                torch.zeros(10),
            )
        img_np = np.array(img)

        if img_np.shape[0] != self.image_size or img_np.shape[1] != self.image_size:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img_np = np.array(img)

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = extract_simple_morphology(img_np)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 127.5 - 1.0
        label_tensor = torch.from_numpy(label).float()

        return img_tensor, label_tensor

    def save_label_cache(self, cache_path: str) -> None:
        logger.info(f"Building label cache for {len(self.keys)} samples...")
        labels = np.zeros((len(self.keys), 10), dtype=np.float32)
        env = self._get_env()

        n_failed = 0
        for i, key in enumerate(self.keys):
            with env.begin(write=False) as txn:
                buf = txn.get(key.encode("utf-8"))
            if buf is None:
                n_failed += 1
                continue
            try:
                img = Image.open(io.BytesIO(buf)).convert("RGB")
            except Exception:
                n_failed += 1
                continue
            img_np = np.array(img)
            if img_np.shape[0] != self.image_size or img_np.shape[1] != self.image_size:
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                img_np = np.array(img)
            labels[i] = extract_simple_morphology(img_np)

            if (i + 1) % 10000 == 0:
                logger.info(f"  Extracted {i + 1}/{len(self.keys)} labels")

        if n_failed > 0:
            logger.warning(f"  {n_failed} entries failed to decode (skipped)")

        np.savez(cache_path, labels=labels, keys=self.keys)
        self.labels = labels
        logger.info(f"Saved label cache to {cache_path}")


def compute_normalization_stats(
    dataset: LMDBMorphDataset,
) -> Tuple[np.ndarray, np.ndarray]:
    if dataset.labels is not None:
        labels = dataset.labels
    else:
        labels = np.array([dataset[i][1].numpy() for i in range(len(dataset))])

    valid_mask = np.any(labels != 0, axis=1)
    valid_labels = labels[valid_mask]

    mean = valid_labels.mean(axis=0).astype(np.float32)
    std = valid_labels.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    return mean, std


def train_morph_proxy(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cache_path = str(output_dir / "morph_labels_cache.npz")
    dataset = LMDBMorphDataset(
        lmdb_path=args.lmdb_path,
        cache_path=cache_path,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )

    if dataset.labels is None:
        dataset.save_label_cache(cache_path)

    label_mean, label_std = compute_normalization_stats(dataset)
    logger.info(f"Label mean: {label_mean}")
    logger.info(f"Label std:  {label_std}")

    np.savez(
        output_dir / "morph_norm_stats.npz",
        mean=label_mean,
        std=label_std,
    )

    n_total = len(dataset)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    train_set, val_set = data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Train: {n_train}, Val: {n_val}")

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MorphProxyEncoder(output_dim=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    label_mean_t = torch.from_numpy(label_mean).to(device)
    label_std_t = torch.from_numpy(label_std).to(device)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training for {args.epochs} epochs")

    best_val_loss = float("inf")
    train_log = []

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            labels_norm = (labels - label_mean_t) / label_std_t

            pred = model(images)
            loss = nn.functional.mse_loss(pred, labels_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / max(n_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_ss_res = 0.0
        val_ss_tot = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                labels_norm = (labels - label_mean_t) / label_std_t

                pred = model(images)
                loss = nn.functional.mse_loss(pred, labels_norm)
                val_loss_sum += loss.item()

                val_ss_res += ((pred - labels_norm) ** 2).sum().item()
                val_ss_tot += ((labels_norm - labels_norm.mean(dim=0)) ** 2).sum().item()
                n_val_batches += 1

        avg_val_loss = val_loss_sum / max(n_val_batches, 1)
        r2 = 1.0 - val_ss_res / max(val_ss_tot, 1e-8)

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "r2": r2,
            "lr": optimizer.param_groups[0]["lr"],
        }
        train_log.append(log_entry)

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train_loss={avg_train_loss:.6f}, "
            f"val_loss={avg_val_loss:.6f}, "
            f"R²={r2:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "output_dim": 10,
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                    "r2": r2,
                    "label_mean": label_mean,
                    "label_std": label_std,
                },
                output_dir / "morph_proxy_encoder.pth",
            )
            logger.info(f"  -> Saved best model (val_loss={avg_val_loss:.6f}, R²={r2:.4f})")

    with open(output_dir / "morph_proxy_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    logger.info(f"Training complete. Best val_loss={best_val_loss:.6f}")
    logger.info(f"Checkpoint: {output_dir / 'morph_proxy_encoder.pth'}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Morphology Proxy Encoder")
    parser.add_argument(
        "--lmdb_path",
        type=str,
        default="/path/to/data/xenium_he_rep1_192px.lmdb",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train_morph_proxy(get_args())
