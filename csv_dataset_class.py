#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:21:24 2025

@author: poulimenos
"""
from torch.utils.data import  Dataset
import pandas as pd
import numpy as np
import os
import torch

class CustomCSVData(Dataset):
    def __init__(self, file_paths, sequence_length=20):
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        self.body_parts = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
                           "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "HEAD"]
        self.xyz_columns = [[f"{part}_{axis}" for axis in "xyz"] for part in self.body_parts]
        
        # Βρίσκουμε όλα τα αρχεία CSV αν δοθεί φάκελος
        self.file_paths = []
        for path in file_paths:
            if os.path.isdir(path):  # Αν είναι φάκελος
                self.file_paths.extend(
                    [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
                )
            elif os.path.isfile(path):  # Αν είναι αρχείο
                self.file_paths.append(path)
        
        # Επεξεργασία δεδομένων από κάθε αρχείο
        for file_path in self.file_paths:
            df = pd.read_csv(file_path)
            features, labels = self.process_patient_data(df)
            
            # Διαχωρισμός σε sliding windows
            for i in range(len(features) - sequence_length + 1):
                window = features[i:i + sequence_length]  # (sequence_length, 3, 9)
                label_window = labels[i:i + sequence_length]  # Όλα τα labels του παραθύρου
                self.data.append(window)
                self.labels.append(label_window)

    def process_patient_data(self, df):
        # Επιλογή μόνο των απαραίτητων στηλών (xyz για κάθε body part)
        xyz_data = []
        for part_columns in self.xyz_columns:
            xyz_data.append(df[part_columns].values)  # Στήλες x, y, z για κάθε part
        
        # Μετατροπή των δεδομένων σε μορφή (samples, 3, 9)
        features = np.stack(xyz_data, axis=-1)  # Συνδυασμός (samples, xyz, parts)
        features = features.transpose(0, 2, 1)  # Αναδιάταξη σε (samples, 3, 9)
        
        # Labels (αν υπάρχουν)
        labels = df["VGRF"].values.astype("float")
        
        return features, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32)  # (sequence_length, 3, 9)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)  # Όλα τα labels του παραθύρου
        return data_tensor, label_tensor
    