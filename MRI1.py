#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np
import pandas as pd
from tqdm import tqdm
import monai
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Resized,
    NormalizeIntensityd, RandFlipd, RandRotate90d, RandShiftIntensityd,
    ToTensord, Compose
)
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import set_determinism
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight


train_transforms = Compose([
    LoadImaged(keys=["image_dwi", "image_dce"]),
    EnsureChannelFirstd(keys=["image_dwi", "image_dce"]),
    Orientationd(keys=["image_dwi", 'image_dce'], axcodes="RAS"),
    Resized(keys=["image_dwi"], spatial_size=(96, 96, 32)),
    Resized(keys=["image_dce"], spatial_size=(96, 96, 32)),
    NormalizeIntensityd(keys=["image_dwi", "image_dce"], nonzero=True, channel_wise=True),
    RandFlipd(keys=["image_dwi"], spatial_axis=[0, 1, 2], prob=0.5),
    RandFlipd(keys=["image_dce"], spatial_axis=[0, 1, 2], prob=0.5),
    RandRotate90d(keys=["image_dwi", 'image_dce'], prob=0.5, max_k=3),
    RandShiftIntensityd(keys=["image_dwi", 'image_dce'], offsets=0.1, prob=0.5),
    ToTensord(keys=['image_dwi', 'image_dce', 'clinical', 'label'])
])

val_transforms = Compose([
    LoadImaged(keys=["image_dwi", 'image_dce']),
    EnsureChannelFirstd(keys=["image_dwi", 'image_dce']),
    Orientationd(keys=["image_dwi", 'image_dce'], axcodes="RAS"),
    Resized(keys=["image_dwi"], spatial_size=(96, 96, 32)),
    Resized(keys=["image_dce"], spatial_size=(96, 96, 32)),
    NormalizeIntensityd(keys=["image_dwi", 'image_dce'], nonzero=True, channel_wise=True),
    ToTensord(keys=['image_dwi', 'image_dce', 'clinical', 'label'])
])


class DataProcessor:
    def __init__(self, csv_path):
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']
        self.df_raw = None
        
        for encoding in encodings:
            try:
                self.df_raw = pd.read_csv(csv_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if self.df_raw is None:
            raise ValueError(f"Cannot read CSV: {csv_path}")
        
        self.df_raw['patient_ID'] = self.df_raw['patient_ID'].astype(str).str.strip()
        
        self.clinical_cols = ['T_stage', 'HER2_status', 'NAC_classification', 
                             'ER_status', 'PR_status', 'Ki_67', 'bpCR']
        
        self.hospital_paths = {
            1: '/path/to/hospital1/roi_crop',
            2: '/path/to/hospital2/roi_crop',
            3: '/path/to/hospital3/roi_crop',
            4: '/path/to/hospital4/roi_crop',
            5: '/path/to/hospital5/roi_crop'
        }

    def split_train_val(self, train_hospitals=[3, 5], val_hospitals=[1, 2, 4]):
        train_dict = []
        val_dict = []
        
        for _, row in self.df_raw.iterrows():
            patient_id = str(row['patient_ID'])
            hospital = int(row['hospital'])
            
            base_path = self.hospital_paths.get(hospital)
            if not base_path:
                continue
            
            dce_path = os.path.join(base_path, f"{patient_id}_c2_roi.nii.gz")
            dwi_path = os.path.join(base_path, f"{patient_id}_dwi2_roi.nii.gz")
            
            if not all(os.path.exists(p) for p in [dce_path, dwi_path]):
                continue
            
            data_item = {
                'image_dwi': dwi_path,
                'image_dce': dce_path,
                'clinical': row[self.clinical_cols[:-1]].tolist(),
                'label': int(row['bpCR']),
                'patient_id': patient_id,
                'hospital': hospital
            }
            
            if hospital in train_hospitals:
                train_dict.append(data_item)
            elif hospital in val_hospitals:
                val_dict.append(data_item)
        
        return train_dict, val_dict

    def get_all_data(self):
        all_data = []
        
        for _, row in self.df_raw.iterrows():
            patient_id = str(row['patient_ID'])
            hospital = int(row['hospital'])
            
            base_path = self.hospital_paths.get(hospital)
            if not base_path:
                continue
            
            dce_path = os.path.join(base_path, f"{patient_id}_c2_roi.nii.gz")
            dwi_path = os.path.join(base_path, f"{patient_id}_dwi2_roi.nii.gz")
            
            if not all(os.path.exists(p) for p in [dce_path, dwi_path]):
                continue
            
            data_item = {
                'image_dwi': dwi_path,
                'image_dce': dce_path,
                'clinical': row[self.clinical_cols[:-1]].tolist(),
                'label': int(row['bpCR']),
                'patient_id': patient_id,
                'hospital': hospital
            }
            all_data.append(data_item)
        
        return all_data


class FeatureQualityEnhancer(nn.Module):
    def __init__(self, input_dim=512, enhanced_dim=256, device=None):
        super().__init__()
        
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.feature_enhancer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, enhanced_dim),
            nn.LayerNorm(enhanced_dim)
        ).to(self.device)
        
        self.discriminative_head = nn.Sequential(
            nn.Linear(enhanced_dim, enhanced_dim // 2),
            nn.ReLU(),
            nn.Linear(enhanced_dim // 2, enhanced_dim)
        ).to(self.device)
        
        self.consistency_head = nn.Sequential(
            nn.Linear(enhanced_dim, enhanced_dim // 4),
            nn.ReLU(),
            nn.Linear(enhanced_dim // 4, enhanced_dim)
        ).to(self.device)
        
    def forward(self, features, labels=None):
        features = features.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        enhanced_features = self.feature_enhancer(features)
        
        if self.training and labels is not None:
            disc_features = self.discriminative_head(enhanced_features)
            consistency_features = self.consistency_head(enhanced_features)
            
            unique_labels = torch.unique(labels)
            intra_class_loss = torch.tensor(0.0, device=self.device)
            inter_class_loss = torch.tensor(0.0, device=self.device)
            consistency_loss = torch.tensor(0.0, device=self.device)
            
            centers = []
            
            for label in unique_labels:
                mask = (labels == label)
                class_features = disc_features[mask]
                class_consistency = consistency_features[mask]
                
                if class_features.shape[0] > 1:
                    center = class_features.mean(0)
                    centers.append(center)
                    
                    intra_class_loss += torch.mean(torch.norm(class_features - center, dim=1))
                    
                    consistency_center = class_consistency.mean(0)
                    consistency_loss += torch.mean(torch.norm(class_consistency - consistency_center, dim=1))
            
            if len(centers) > 1:
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        inter_class_loss -= torch.norm(centers[i] - centers[j])
            
            return enhanced_features, intra_class_loss, inter_class_loss, consistency_loss
        
        return enhanced_features


class DoubleTower(nn.Module):
    def __init__(self, pretrained_dce='', pretrained_dwi='', device=torch.device("cuda:0"), 
                 num_classes=2, fc_hidden_size=256):
        super().__init__()
        
        self.fc_hidden_size = fc_hidden_size
        self.num_classes = num_classes
        self.device = device

        self.model_dce = monai.networks.nets.resnet18(
            spatial_dims=3, n_input_channels=1, num_classes=2, feed_forward=False
        ).to(self.device)
        
        self.model_dwi = monai.networks.nets.resnet18(
            spatial_dims=3, n_input_channels=1, num_classes=2, feed_forward=False
        ).to(self.device)

        if pretrained_dce != '':
            dce_dict = self.model_dce.state_dict()
            dce_pretrain = torch.load(pretrained_dce, map_location=self.device)
            dce_pretrain_dict = {k: v for k, v in dce_pretrain.items() if k in dce_dict.keys()}
            dce_dict.update(dce_pretrain_dict)
            self.model_dce.load_state_dict(dce_dict)

        if pretrained_dwi != '':
            dwi_dict = self.model_dwi.state_dict()
            dwi_pretrain = torch.load(pretrained_dwi, map_location=self.device)
            dwi_pretrain_dict = {k: v for k, v in dwi_pretrain.items() if k in dwi_dict.keys()}
            dwi_dict.update(dwi_pretrain_dict)
            self.model_dwi.load_state_dict(dwi_dict)

        self.attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True, device=self.device)
        
        self.feature_enhancer = FeatureQualityEnhancer(512, 256, device=self.device)

        self.Linear1 = Linear(256, self.fc_hidden_size, device=self.device)
        self.Linear2 = Linear(self.fc_hidden_size, self.num_classes, device=self.device)
        self.dropout = nn.Dropout(0.2)

    def extract_features(self, x1, x2):
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        
        encode_output1 = self.model_dce(x1)
        encode_output2 = self.model_dwi(x2)
        concatenated = encode_output1 * encode_output2
        
        concatenated = concatenated.unsqueeze(1)
        attn_output, _ = self.attn(concatenated, concatenated, concatenated)
        attn_output = attn_output.squeeze(1)

        enhanced_features = self.feature_enhancer(attn_output)
        fc1 = self.Linear1(enhanced_features)
        return fc1

    def forward(self, x1, x2, structured_data):
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        structured_data = structured_data.to(self.device)
        
        encode_output1 = self.model_dce(x1)
        encode_output2 = self.model_dwi(x2)
        concatenated = encode_output1 * encode_output2
        
        concatenated = concatenated.unsqueeze(1)
        attn_output, _ = self.attn(concatenated, concatenated, concatenated)
        attn_output = attn_output.squeeze(1)

        enhanced_features = self.feature_enhancer(attn_output)
        fc1 = F.relu(self.Linear1(enhanced_features))
        fc1 = self.dropout(fc1)
        fc2 = self.Linear2(torch.concat([fc1], dim=-1))
        
        return F.log_softmax(fc2, dim=-1)

    def forward_with_feature_loss(self, x1, x2, structured_data, labels):
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        structured_data = structured_data.to(self.device)
        labels = labels.to(self.device)
        
        encode_output1 = self.model_dce(x1)
        encode_output2 = self.model_dwi(x2)
        concatenated = encode_output1 * encode_output2
        
        concatenated = concatenated.unsqueeze(1)
        attn_output, _ = self.attn(concatenated, concatenated, concatenated)
        attn_output = attn_output.squeeze(1)

        enhanced_features, intra_loss, inter_loss, consistency_loss = self.feature_enhancer(
            attn_output, labels
        )

        fc1 = F.relu(self.Linear1(enhanced_features))
        fc1 = self.dropout(fc1)
        fc2 = self.Linear2(torch.concat([fc1], dim=-1))
        outputs = F.log_softmax(fc2, dim=-1)
        
        return outputs, intra_loss, inter_loss, consistency_loss


def train_model(csv_path, model_save_path="./best_model.pth", max_epochs=150):
    print("Training DoubleTower with Feature Quality Enhancer")
    print("=" * 80)
    
    set_determinism(seed=42)
    
    processor = DataProcessor(csv_path)
    train_data, val_data = processor.split_train_val(train_hospitals=[3, 5], val_hospitals=[1, 2, 4])
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    if len(train_data) == 0 or len(val_data) == 0:
        print("ERROR: Empty dataset")
        return False

    train_labels = [item['label'] for item in train_data]
    val_labels = [item['label'] for item in val_data]
    print(f"Train labels: 0={train_labels.count(0)}, 1={train_labels.count(1)}")
    print(f"Val labels: 0={val_labels.count(0)}, 1={val_labels.count(1)}")

    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0, num_workers=4)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = DoubleTower(device=device)
    model = model.to(device)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)

    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    best_metric = -1
    best_metric_epoch = -1
    patience = 20
    patience_counter = 0

    print(f"Training for {max_epochs} epochs...\n")

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        
        model.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_feature_loss = 0
        step = 0
        
        for batch_data in tqdm(train_loader, desc="Training", leave=False):
            step += 1
            
            input_dce = batch_data["image_dce"].to(device)
            input_dwi = batch_data["image_dwi"].to(device)
            input_clinical = batch_data["clinical"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            
            outputs, intra_loss, inter_loss, consistency_loss = model.forward_with_feature_loss(
                input_dce, input_dwi, input_clinical, labels
            )
            
            classification_loss = loss_function(outputs, labels)
            
            lambda_intra = 0.1
            lambda_inter = 0.05
            lambda_consistency = 0.05
            
            feature_loss = (lambda_intra * intra_loss + 
                          lambda_inter * inter_loss + 
                          lambda_consistency * consistency_loss)
            
            total_loss = classification_loss + feature_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_cls_loss += classification_loss.item()
            epoch_feature_loss += feature_loss.item()

        epoch_loss /= step
        epoch_cls_loss /= step
        epoch_feature_loss /= step
        
        print(f"Loss: {epoch_loss:.4f} (cls: {epoch_cls_loss:.4f}, feat: {epoch_feature_loss:.4f})")
        
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                val_dce = batch_data["image_dce"].to(device)
                val_dwi = batch_data["image_dwi"].to(device)
                val_clinical = batch_data["clinical"].to(device)
                labels = batch_data["label"].to(device)
                
                outputs = model(val_dce, val_dwi, val_clinical)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        scheduler.step(val_loss)
        
        if val_accuracy > best_metric:
            best_metric = val_accuracy
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print("Model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping")
            break

    print(f"\nTraining completed!")
    print(f"Best val accuracy: {best_metric:.4f} (Epoch {best_metric_epoch})")
    print(f"Model saved: {model_save_path}")
    
    return True


def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    patient_ids = []
    
    print("Extracting features...")
    
    with torch.no_grad():
        for batch_data in tqdm(data_loader, leave=False):
            input_dce = batch_data["image_dce"].to(device)
            input_dwi = batch_data["image_dwi"].to(device)
            
            batch_features = model.extract_features(input_dce, input_dwi)
            
            features.append(batch_features.cpu().numpy())
            labels.append(batch_data["label"].numpy())
            
            if "patient_id" in batch_data:
                patient_ids.extend(batch_data["patient_id"])
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    return features, labels, patient_ids


def extract_features_main(csv_path, model_path="./best_model.pth", output_dir="./extracted_features"):
    print("Extracting enhanced features...")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    processor = DataProcessor(csv_path)
    all_data = processor.get_all_data()
    print(f"Found {len(all_data)} samples")
    
    if len(all_data) == 0:
        print("ERROR: No data")
        return False
    
    dataset = Dataset(data=all_data, transform=val_transforms)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    model = DoubleTower(device=device)
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded: {model_path}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False
    
    features, labels, patient_ids = extract_features(model, data_loader, device)
    
    print(f"Extracted: {features.shape[0]} samples, {features.shape[1]} features")
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "features.npy"), features)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "patient_ids.npy"), np.array(patient_ids))
    
    features_df = pd.DataFrame(features)
    features_df.columns = [f"feature_{i}" for i in range(features.shape[1])]
    features_df["patient_id"] = patient_ids
    features_df["label"] = labels
    features_df.to_csv(os.path.join(output_dir, "features.csv"), index=False)
    
    print(f"Features saved to: {output_dir}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DoubleTower with Feature Quality Enhancer')
    parser.add_argument('--mode', choices=['train', 'extract', 'both'], default='both')
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--model_path', default='./best_model.pth')
    parser.add_argument('--output_dir', default='./extracted_features')
    parser.add_argument('--max_epochs', type=int, default=150)
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        train_model(args.csv_path, args.model_path, args.max_epochs)
    
    if args.mode in ['extract', 'both']:
        extract_features_main(args.csv_path, args.model_path, args.output_dir)