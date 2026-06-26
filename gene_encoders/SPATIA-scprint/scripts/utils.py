import os
import random

import numpy as np
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def loader_construction(data_path):
    data = sc.read_h5ad(data_path)
    X_all = data.X
    y_all = data.obs.values[:,0]
    input_dim = X_all.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=1)
    train_set = CellDataset(X_train, y_train)
    test_set = CellDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=0)
    return train_loader, test_loader, input_dim


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def evaluate(y_true, y_pred):
    acc= cluster_acc(y_true, y_pred)
    f1=0
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    return acc, f1, nmi, ari, homo, comp


tissue_mapping = {
    "ovary": "UBERON:0006960",
    "prostate": "UBERON:0004243",
    "brain": "UBERON:0003714",
    "lung": "UBERON:0003615",
    "colon": "UBERON:0001155",
    "tonsil": "UBERON:0002372",
    "lymph node": "UBERON:0000029",
    "heart": "UBERON:0000948",
    "pancreas": "UBERON:0001264",
    "skin": "UBERON:0002097",
    "bone": "UBERON:0002481",
    "liver": "UBERON:0002107",
    "bone marrow": "UBERON:0002371",
    "kidney": "UBERON:0002113",
    "unknown": "UBERON:0002113",
    "breast": "UBERON:0000310",
    "cervical": "UBERON:0016404",
    "colorectal": "UBERON:0012652",
}
disease_mapping = {
    "melanoma": "MONDO:0005105",
    "alzheimers": "MONDO:0004975",
    "ductal adenocarcinoma": "MONDO:0004970",
    "invasive ductal carcinoma": "MONDO:0004953",
    "follicular lymphoid hyperplasia": "MONDO:0005043",
    "reactive follicular hyperplasia": "MONDO:0005043",
    "cancer": "MONDO:0004992",
    "healthy": "PATO:0000461",
    "acute lymphoid leukemia": "MONDO:0004967",
    "unknown": "PATO:0000461",
    "invasive lobular carcinoma": "MONDO:0005051",
    "glioblastoma": "MONDO:0018177",
    "reactive hyperplasia": "MONDO:0005043",
}

cell_mapping = {
    "ovary": "CL:0002095",
    "prostate": "UBERON:0004243",
    "brain": "UBERON:0003714",
    "lung": "UBERON:0003615",
    "colon": "UBERON:0001155",
    "tonsil": "UBERON:0002372",
    "lymph node": "UBERON:0000029",
    "heart": "UBERON:0000948",
    "pancreas": "UBERON:0001264",
    "skin": "CL:0002320",
    "bone": "UBERON:0002481",
    "liver": "UBERON:0002107",
    "bone marrow": "UBERON:0002371",
    "kidney": "UBERON:0002113",
    "unknown": "unknown",
    "breast": "UBERON:0000310",
    "cervical": "UBERON:0016404",
    "colorectal": "UBERON:0012652",
}
