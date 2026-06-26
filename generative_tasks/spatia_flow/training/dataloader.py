import os
from pathlib import Path
import pdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from training.data_utils import CustomTransform, read_files_batch, read_files_pert

class CellDataset:
    
    def __init__(self, args, device):

        assert os.path.exists(args.image_path), 'The data path does not exist'
        assert os.path.exists(args.data_index_path), 'The data index path does not exist'

        self.image_path = args.image_path
        self.data_index_path = args.data_index_path
        self.embedding_path = args.embedding_path
        self.augment_train = args.augment_train
        self.normalize = args.normalize
        self.mol_list = args.mol_list
        self.ood_set = args.ood_set
        self.trainable_emb = args.batch_correction
        self.dataset_name = args.dataset_name

        self.batch_correction = args.batch_correction
        self.multimodal = args.multimodal
        if self.trainable_emb or self.batch_correction:
            self.latent_dim = args.latent_dim

        if not self.batch_correction:
            self.add_controls = args.add_controls
            self.batch_key = None
        else:
            self.add_controls = None
            self.batch_key = args.batch_key

        self.device = device 

        self.fold_datasets = self._read_folds()

        self.y_names = np.unique(self.fold_datasets['train']["ANNOT"])

        self._initialize_mol_names()

        self.y2id = {y: id for id, y in enumerate(self.y_names)}
        self.n_y = len(self.y_names)
        self.iter_ctrl = args.iter_ctrl
        self.initialize_embeddings()

        self.fold_datasets = {
            'train': CellDatasetFold('train', 
                                     self.image_path, 
                                     self.fold_datasets['train'],
                                     self.mol2id,
                                     self.y2id, 
                                     self.augment_train, 
                                     self.normalize,
                                     dataset_name=self.dataset_name,
                                     add_controls=self.add_controls, 
                                     batch_correction=self.batch_correction,
                                     batch_key=self.batch_key,
                                     multimodal=self.multimodal,
                                     cpd_name=self.cpd_name,
                                     iter_ctrl=self.iter_ctrl),
            
            'test': CellDatasetFold('test',
                                    self.image_path,
                                    self.fold_datasets['test'],
                                    self.mol2id, 
                                    self.y2id, 
                                    self.augment_train, 
                                    self.normalize,
                                    dataset_name=self.dataset_name,
                                    add_controls=self.add_controls,
                                    batch_correction=self.batch_correction,
                                    batch_key=self.batch_key,
                                    multimodal=self.multimodal,
                                    cpd_name=self.cpd_name,
                                    iter_ctrl=False)}

    def _read_folds(self):
        dataset = pd.read_csv(self.data_index_path, index_col=0)
        
        self.cpd_name = "BROAD_SAMPLE" if self.dataset_name == "cpg0000" else "CPD_NAME"

        if self.mol_list:
            dataset = dataset.loc[dataset[self.cpd_name].isin(self.mol_list)]
        if self.ood_set is not None:
            dataset = dataset.loc[~dataset[self.cpd_name].isin(self.ood_set)]
        
        dataset_splits = dict()
        
        for fold_name in ['train', 'test']:
            dataset_splits[fold_name] = {}
            
            subset = dataset.loc[dataset.SPLIT == fold_name]
            for key in subset.columns:
                dataset_splits[fold_name][key] = np.array(subset[key])
            if not self.batch_correction:
                if self.dataset_name == 'bbbc021':
                    if not self.add_controls:
                        dataset_splits[fold_name]["trt_idx"] = (dataset_splits[fold_name]["STATE"] == 1)
                    else:
                        dataset_splits[fold_name]["trt_idx"] = (np.isin(dataset_splits[fold_name]["STATE"], ["trt", "control"]))
                    dataset_splits[fold_name]["ctrl_idx"] = (dataset_splits[fold_name]["STATE"] == 0)
                elif self.dataset_name == "rxrx1":
                    assert not self.add_controls, "Controls are not supported for rxrx1 dataset."
                    dataset_splits[fold_name]["trt_idx"] = (dataset_splits[fold_name]["ANNOT"] == "treated")
                    dataset_splits[fold_name]["ctrl_idx"] = (dataset_splits[fold_name]["ANNOT"] == "negative_control")
                elif self.dataset_name == "cpg0000":
                    assert not self.add_controls, "Controls are not supported for cpg0000 dataset."
                    dataset_splits[fold_name]["trt_idx"] = (dataset_splits[fold_name]["STATE"] == "trt")
                    dataset_splits[fold_name]["ctrl_idx"] = (dataset_splits[fold_name]["STATE"] == "control")
        return dataset_splits

    def _initialize_mol_names(self):
        if not self.batch_correction:
            if not self.multimodal:
                if self.add_controls:
                    self.mol_names = np.unique(self.fold_datasets["train"][self.cpd_name])
                else:
                    self.mol_names = np.unique(self.fold_datasets["train"][self.cpd_name][self.fold_datasets["train"]["trt_idx"]])
                self.n_mol = len(self.mol_names)
            else:
                self.mol_names = {}
                for pert_type in self.y_names:
                    idx_pert = self.fold_datasets["train"]["ANNOT"] == pert_type
                    if self.add_controls:
                        self.mol_names[pert_type] = np.unique(self.fold_datasets["train"][self.cpd_name][idx_pert])
                    else:
                        trt_idx = self.fold_datasets["train"]["trt_idx"][idx_pert]
                        self.mol_names[pert_type] = np.unique(self.fold_datasets["train"][self.cpd_name][idx_pert][trt_idx])
                self.n_mol = {key: len(val) for key, val in self.mol_names.items()} 
        else: 
            self.mol_names = np.unique(self.fold_datasets['train'][self.batch_key])
            self.n_mol = len(self.mol_names)

    def initialize_embeddings(self):
        if self.multimodal and (not self.trainable_emb and not self.batch_correction):
            embedding_matrix = []
            mol2id = {}
            self.latent_dim = {}

            for mod in self.y_names:
                embedding_matrix_modality = pd.read_csv(self.embedding_path[mod], index_col=0)
                embedding_matrix_modality = embedding_matrix_modality.loc[self.mol_names[mod]]
                embedding_matrix_modality = torch.tensor(embedding_matrix_modality.values, dtype=torch.float32, device=self.device)
                self.latent_dim[mod] = embedding_matrix_modality.shape[1]
                embedding_matrix_modality = torch.nn.Embedding.from_pretrained(embedding_matrix_modality, freeze=True).to(self.device)
                embedding_matrix.append(embedding_matrix_modality)
                mol2id[mod] = {mol: id for id, mol in enumerate(self.mol_names[mod])}
                
            self.embedding_matrix = torch.nn.ModuleList(embedding_matrix)
            self.mol2id = mol2id
            
        else:
            if self.trainable_emb or self.batch_correction:
                self.latent_dim = self.latent_dim
                self.embedding_matrix = torch.nn.Embedding(self.n_mol, self.latent_dim).to(self.device).to(torch.float32)
            else:
                embedding_matrix = pd.read_csv(self.embedding_path, index_col=0)
                embedding_matrix = embedding_matrix.loc[self.mol_names]
                embedding_matrix = torch.tensor(embedding_matrix.values, dtype=torch.float32, device=self.device)
                self.embedding_matrix = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=True).to(self.device)
            
                self.latent_dim = embedding_matrix.shape[1]
            
            self.mol2id = {mol: id for id, mol in enumerate(self.mol_names)}

class CellDatasetFold(Dataset):
    
    def __init__(self,
                 fold, 
                 image_path, 
                 data, 
                 mol2id,
                 y2id,
                 augment_train=True, 
                 normalize=False, 
                 dataset_name="bbbc021", 
                 add_controls=None,
                 batch_correction=False, 
                 batch_key="BATCH", 
                 multimodal=False, 
                 cpd_name="CPD_NAME",
                 iter_ctrl=False):
        super(CellDatasetFold, self).__init__()

        self.image_path = image_path
        self.fold = fold  
        self.data = data
        self.dataset_name = dataset_name
        self.add_controls = add_controls
        self.batch_correction = batch_correction
        self.multimodal = multimodal
        self.cpd_name = cpd_name
        self.iter_ctrl = iter_ctrl
        if self.batch_correction:
            self.file_names = data['SAMPLE_KEY']
            self.mols = data[batch_key]
            self.y = data['ANNOT']
            if dataset_name == "bbbc021":
                self.dose = data['DOSE']
            else:
                self.dose = None
        else:
            self.file_names = {}
            self.mols = {}
            self.y = {}
            self.batch = {}
            if dataset_name == "bbbc021":
                self.dose = {}
            else:
                self.dose = None
            
            for cond in ["ctrl", "trt"]:
                if cond == "trt" and add_controls:
                    self.file_names[cond] = self.data['SAMPLE_KEY']
                    self.mols[cond] = self.data[cpd_name]
                    self.y[cond] = self.data['ANNOT']
                    if dataset_name == "bbbc021":
                        self.dose[cond] = self.data['DOSE']
                else:
                    self.file_names[cond] = self.data['SAMPLE_KEY'][self.data[f"{cond}_idx"]]
                    self.mols[cond] = self.data[cpd_name][self.data[f"{cond}_idx"]]
                    self.y[cond] = self.data['ANNOT'][self.data[f"{cond}_idx"]]
                    batch_key = "PLATE" if dataset_name == "cpg0000" else "BATCH"
                    self.batch[cond] = self.data[batch_key][self.data[f"{cond}_idx"]]
                    if dataset_name == "bbbc021":
                        self.dose[cond] = self.data['DOSE'][self.data[f"{cond}_idx"]]
        del data 
        
        self.augment_train = augment_train
        
        self.mol2id = mol2id
        self.y2id = y2id
        
        self.transform = CustomTransform(augment=(self.augment_train and self.fold == 'train'), normalize=normalize)

        
    def __len__(self):
        if self.batch_correction:
            return len(self.file_names)
        else:
            return len(self.file_names["ctrl"]) if self.iter_ctrl else len(self.file_names["trt"])

    def __getitem__(self, idx):
        if self.batch_correction:
            return read_files_batch(self.file_names, 
                                    self.mols,
                                    self.mol2id,
                                    self.y2id, 
                                    self.y, 
                                    self.transform,
                                    self.image_path, 
                                    self.dataset_name, 
                                    idx)
        else:
            return read_files_pert(self.file_names, 
                                   self.mols, 
                                   self.mol2id, 
                                   self.y2id, 
                                   self.dose, 
                                   self.y, 
                                   self.transform, 
                                   self.image_path, 
                                   self.dataset_name,
                                   idx,
                                   self.multimodal,
                                   self.batch,
                                   self.iter_ctrl,)

class CellDataLoader(LightningDataModule):
    
    def __init__(self, args):

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.init_dataset()

    def init_dataset(self):
        self.training_set, self.test_set = self.create_torch_datasets()
        sampler_train = torch.utils.data.DistributedSampler(
            self.training_set, num_replicas=self.args.num_tasks, rank=self.args.global_rank, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            self.test_set, num_replicas=self.args.num_tasks, rank=self.args.global_rank, shuffle=False
        )
        self.loader_train = torch.utils.data.DataLoader(self.training_set, 
                                                        sampler=sampler_train,
                                                        batch_size=self.args.batch_size, 
                                                        num_workers=self.args.num_workers, 
                                                        pin_memory=self.args.pin_mem,
                                                        drop_last=True)  
        self.loader_test = torch.utils.data.DataLoader(self.test_set, 
                                                       sampler=sampler_test,
                                                       batch_size=self.args.batch_size, 
                                                       num_workers=self.args.num_workers, 
                                                       drop_last=False)          

    def create_torch_datasets(self):
        dataset = CellDataset(self.args, device=self.device) 
        
        self.dim = self.args.n_channels

        self.embedding_matrix = dataset.embedding_matrix  
        self.latent_dim = dataset.latent_dim

        self.n_mol = dataset.n_mol
        self.num_y = dataset.n_y 

        training_set, test_set = dataset.fold_datasets.values()  
        
        self.mol2id = dataset.mol2id
        self.y2id = dataset.y2id
        if self.args.multimodal:
            self.id2mol = {}
            self.id2y = {}
            for mod in self.mol2id:
                self.id2mol[mod] = {val:key for key,val in self.mol2id[mod].items()}
                self.id2y[mod] = {val:key for key,val in self.y2id.items()} 
        else:
            self.id2mol = {val:key for key,val in self.mol2id.items()}
            self.id2y = {val:key for key,val in self.y2id.items()} 

        del dataset
        return training_set, test_set
    
    def train_dataloader(self):
        return self.loader_train
    
    def val_dataloader(self):
        return self.loader_test
    
    def test_dataloader(self):
        return self.loader_test

class CellDataLoader_Eval(LightningDataModule):
    
    def __init__(self, args):

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.init_dataset()

    def init_dataset(self):
        self.training_set, self.test_set = self.create_torch_datasets()
        self.loader_train = torch.utils.data.DataLoader(self.training_set, 
                                                        shuffle=True,
                                                        batch_size=self.args.batch_size, 
                                                        num_workers=self.args.num_workers, 
                                                        pin_memory=self.args.pin_mem,
                                                        drop_last=True)  
        self.loader_test = torch.utils.data.DataLoader(self.test_set, 
                                                       shuffle=False,
                                                       batch_size=self.args.batch_size, 
                                                       num_workers=self.args.num_workers, 
                                                       drop_last=False)          

    def create_torch_datasets(self):
        dataset = CellDataset(self.args, device=self.device) 
        
        self.dim = self.args.n_channels

        self.embedding_matrix = dataset.embedding_matrix  
        self.latent_dim = dataset.latent_dim

        self.n_mol = dataset.n_mol
        self.num_y = dataset.n_y 

        training_set, test_set = dataset.fold_datasets.values()  
        
        self.mol2id = dataset.mol2id
        self.y2id = dataset.y2id
        if self.args.multimodal:
            self.id2mol = {}
            self.id2y = {}
            for mod in self.mol2id:
                self.id2mol[mod] = {val:key for key,val in self.mol2id[mod].items()}
                self.id2y[mod] = {val:key for key,val in self.y2id.items()} 
        else:
            self.id2mol = {val:key for key,val in self.mol2id.items()}
            self.id2y = {val:key for key,val in self.y2id.items()} 

        del dataset
        return training_set, test_set
    
    def train_dataloader(self):
        return self.loader_train
    
    def val_dataloader(self):
        return self.loader_test
    
    def test_dataloader(self):
        return self.loader_test
