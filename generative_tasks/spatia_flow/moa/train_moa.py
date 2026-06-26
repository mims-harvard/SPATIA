import argparse
from collections import defaultdict
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.dataloader import CellDataLoader_Eval
import torchvision.transforms as T
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from sklearn.metrics import f1_score

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

class MOAClassifier(nn.Module):
    def __init__(self, num_classes, device):
        super(MOAClassifier, self).__init__()

        self.feature_extractor = FrechetInceptionDistance(normalize=True).to(device=device, non_blocking=True)
        for param in self.feature_extractor.inception.parameters():
            param.requires_grad = False


        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        ).to(device)

    def forward(self, x):
        features = (x * 255).byte()
        features = self.feature_extractor.inception(features)
        outputs = self.classifier(features)
        return outputs

def save_checkpoint(model, optimizer, epoch, save_path):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(state, save_path)
    print(f"Checkpoint saved at {save_path}")

def load_checkpoint(model, optimizer, load_path, device):
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded from {load_path}, starting from epoch {start_epoch}")
    return start_epoch

def read_img_from_path(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
    return img

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10, save_path="checkpoint_ood.pth"):
    model.to(device)
    start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x_real_ctrl, x_real_trt = batch['X']
            images = torch.clamp(x_real_trt * 0.5 + 0.5, min=0.0, max=1.0).to(device)
            labels = batch['y_id'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100.*correct/total:.2f}%")

        save_checkpoint(model, optimizer, epoch, save_path)

def evaluate_model(model, dataloader, device, id2y):
    model.eval()
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x_real_ctrl, x_real_trt = batch['X']
            images = torch.clamp(x_real_trt * 0.5 + 0.5, min=0.0, max=1.0).to(device)
            labels = batch['y_id'].to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())


            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    print(f"Test Accuracy: {100. * correct / total:.2f}%")
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Macro-F1 Score: {macro_f1:.4f}")
    print(f"Weighted-F1 Score: {weighted_f1:.4f}")

    print("\nPer-Class Accuracy:")
    for class_id in class_total:
        acc = 100. * class_correct[class_id] / class_total[class_id]
        print(f"Class {id2y[class_id]}: {acc:.2f}%, Total: {class_total[class_id]}")
    
def evaluate_generated_image(model, dataloader, device, img_root_path, id2mol, id2y):
    model.eval()
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            x_real_ctrl, x_real_trt = batch['X']
            y_trg = batch['mols']
            idx_ctrl, idx_trt = batch['idx_ctrl'], batch['idx_trt']
            img_file_ctrl, img_file_trt = batch['file_names']
            labels = batch['y_id'].to(device)
            target_classes = [id2mol[y.item()] for y in y_trg]
            synthetic_samples = []

            for i in range(x_real_ctrl.shape[0]):
                target_class = target_classes[i]
                synthetic_sample = read_img_from_path(os.path.join(img_root_path, target_class + f'/{img_file_trt[i]}.png'))
                synthetic_samples.append(synthetic_sample)

            synthetic_samples = torch.stack(synthetic_samples).to(device).float() / 255.0

            outputs = model(synthetic_samples)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())


            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

            if total >= 5120:
                break

    print(f"Test Generated Image from: {img_root_path}")
    print(f"Overall Accuracy: {100. * correct / total:.2f}%")

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Macro-F1 Score: {macro_f1:.4f}")
    print(f"Weighted-F1 Score: {weighted_f1:.4f}")

    print("\nPer-Class Accuracy:")
    for class_id in class_total:
        acc = 100. * class_correct[class_id] / class_total[class_id]
        print(f"Class {id2y[class_id]}: {acc:.2f}%, Total: {class_total[class_id]}")
    

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datamodule = CellDataLoader_Eval(args)
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    num_classes = datamodule.num_y

    model = MOAClassifier(num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch = 0
    if Path(args.ckpt_path).exists():
        start_epoch = load_checkpoint(model, optimizer, args.ckpt_path, device)
    id2mol = {v: k for k, v in datamodule.mol2id.items()}
    id2y = datamodule.id2y
    if args.mode == 'train':
        train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, save_path=args.ckpt_path)

        evaluate_model(model, test_loader, device, id2y)
    elif args.mode == 'eval':
        assert args.img_root_path is not None, "Image root path is required for evaluation"
        evaluate_generated_image(model, test_loader, device, args.img_root_path, id2mol, id2y)


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root_path', type=str, default=None, help='Image root for results')
    parser.add_argument('--ckpt_path', type=str, default='checkpoint.pth', help='Model path')
    parser.add_argument('--mode', type=str, default='eval', help='Mode: eval or train')
    parser.add_argument('--config_path', type=str, default='../configs/bbbc021_all.yaml', help='Config path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--iter_ctrl', type=bool, default=False, help='Iter ctrl')
    parser.add_argument('--pin_mem', type=bool, default=True, help='Pin mem')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers')
    cli_args = parser.parse_args()
    yaml_config = load_yaml_config(cli_args.config_path)
    yaml_config.update(vars(cli_args))
    args = SimpleNamespace(**yaml_config)
    main(args)
