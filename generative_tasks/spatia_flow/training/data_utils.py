import torch
import torchvision.transforms as T
import numpy as np
from pathlib import Path

class CustomTransform:
    
    def __init__(self, augment=False, normalize=False, dim=0):
        self.augment = augment 
        self.normalize = normalize 
        self.dim = dim
        
    def __call__(self, X):
        random_noise = torch.rand_like(X)
        X = (X + random_noise) / 255.0
        
        t = []
        if self.normalize:
            num_channels = X.shape[self.dim]
            mean = [0.5] * num_channels
            std = [0.5] * num_channels
            t.append(T.Normalize(mean=mean, std=std))
        
        if self.augment:
            t.append(T.RandomHorizontalFlip(p=0.3))
            t.append(T.RandomVerticalFlip(p=0.3))

        trans = T.Compose(t)
        return trans(X)

def read_files_pert(file_names, mols, mol2id, y2id, dose, y, transform, image_path, dataset_name, idx, multimodal, batch, iter_ctrl):
    if iter_ctrl:
        img_file_ctrl = file_names["ctrl"][idx]
        idx_trt = np.random.randint(0, len(file_names["trt"]))
        img_file_trt = file_names["trt"][idx_trt]
        idx_ctrl = idx
    
    else: 
        idx_trt = idx
        img_file_trt = file_names["trt"][idx_trt]
        batch_trt = batch["trt"][idx_trt]

        ctrl_indices_same_batch = np.where(batch["ctrl"] == batch_trt)[0]
        if len(ctrl_indices_same_batch) == 0:
            raise ValueError(f"No control samples found in the same batch as the treated sample (batch: {batch_trt}).")

        idx_ctrl = np.random.choice(ctrl_indices_same_batch)
        img_file_ctrl = file_names["ctrl"][idx_ctrl]

    file_split_ctrl = img_file_ctrl.split('-')
    file_split_trt = img_file_trt.split('-')
    
    if len(file_split_ctrl) > 1:
        file_split_ctrl = file_split_ctrl[1].split("_")
        file_split_trt = file_split_trt[1].split("_")
        path_ctrl = Path(image_path) / "_".join(file_split_ctrl[:2]) / file_split_ctrl[2]
        path_trt = Path(image_path) / "_".join(file_split_trt[:2]) / file_split_trt[2]
        file_ctrl = '_'.join(file_split_ctrl[3:]) + ".npy"
        file_trt = '_'.join(file_split_trt[3:]) + ".npy"
    else:
        file_split_ctrl = file_split_ctrl[0].split("_")
        file_split_trt = file_split_trt[0].split("_")
        if dataset_name == "cpg0000":
            path_ctrl = Path(image_path) / file_split_ctrl[0] / f"{file_split_ctrl[1]}_{file_split_ctrl[2]}"
            path_trt = Path(image_path) / file_split_trt[0] / f"{file_split_trt[1]}_{file_split_trt[2]}"
            file_ctrl = '_'.join(file_split_ctrl[1:]) + ".npy"
            file_trt = '_'.join(file_split_trt[1:]) + ".npy"
        elif dataset_name == "bbbc021":
            path_ctrl = Path(image_path) / file_split_ctrl[0] / f"{file_split_ctrl[1]}"
            path_trt = Path(image_path) / file_split_trt[0] / f"{file_split_trt[1]}"
            file_ctrl = '_'.join(file_split_ctrl[2:]) + ".npy"
            file_trt = '_'.join(file_split_trt[2:]) + ".npy"
        
    img_ctrl, img_trt = np.load(path_ctrl / file_ctrl), np.load(path_trt / file_trt)
    img_ctrl, img_trt = torch.from_numpy(img_ctrl).float(), torch.from_numpy(img_trt).float()
    img_ctrl, img_trt = img_ctrl.permute(2, 0, 1), img_trt.permute(2, 0, 1)
    img_ctrl, img_trt = transform(img_ctrl), transform(img_trt)
    
    if multimodal:
        y_mod = y["trt"][idx_trt]
        mol = mol2id[y_mod][mols["trt"][idx_trt]]
    else:
        mol = mol2id[mols["trt"][idx_trt]]
    
    return {
        'X': (img_ctrl, img_trt),
        'mols': mol,
        'y_id': y2id[y["trt"][idx_trt]],
        'dose': dose["trt"][idx_trt],
        'file_names': (img_file_ctrl, img_file_trt),
        'idx_trt': idx_trt,
        'idx_ctrl': idx_ctrl,
        'batch': batch_trt,
    } if dataset_name == "bbbc021" else {
        'X': (img_ctrl, img_trt),
        'mols': mol,
        'y_id': y2id[y["trt"][idx_trt]],
        'file_names': (img_file_ctrl, img_file_trt),
        'idx_trt': idx_trt,
        'idx_ctrl': idx_ctrl,
        'batch': batch_trt,
    }

def read_files_batch(file_names, mols, mol2id, y2id, y, transform, image_path, dataset_name, idx):
    img_file = file_names[idx]
    file_split = img_file.split('-')
    
    if dataset_name == "rxrx1":
        file_split = file_split[1].split("_")
        path = Path(image_path) / "_".join(file_split[:2]) / file_split[2]
        file = '_'.join(file_split[3:]) + ".npy"
    elif dataset_name in ["bbbc021", "bbbc025"]:
        file_split = file_split[0].split("_")
        path = Path(image_path) / file_split[0] / file_split[1]
        file = '_'.join(file_split[2:]) + ".npy"
    else:
        file_split = file_split[0].split("_")
        path = Path(image_path) / file_split[0] / f"{file_split[1]}_{file_split[2]}"
        file = '_'.join(file_split[1:]) + ".npy"
        
    img = np.load(path / file)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)
    img = transform(img)

    mol = mol2id[mols[idx]]
    
    return {
        'X': img,
        'mols': mol,
        'y_id': y2id[y[idx]],
        'file_names': img_file
    }

def convert_6ch_to_3ch(images):
    weights = torch.tensor([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ], dtype=images.dtype, device=images.device)
    
    images_rgb = torch.einsum('bchw,cn->bnhw', images, weights)
    
    images_rgb = torch.clamp(images_rgb, -1, 1)
    
    return images_rgb

def convert_5ch_to_3ch(images):
    images_rgb = images[:, :3, :, :]
    return images_rgb