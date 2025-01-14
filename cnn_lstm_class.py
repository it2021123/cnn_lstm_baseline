#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:18:03 2025

@author: poulimenos
"""
import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, hidden_dim, lstm_layers, dropout_rate=0.1):
        super(CNNLSTM, self).__init__()

        # CNN για επεξεργασία των χαρακτηριστικών
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 3), padding=(0, 1)),  # Συμβατό με Features=9
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Spatial dimension μειώνεται
            nn.Dropout(dropout_rate),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate),
        )

        # Υπολογισμός Flattened Features από το CNN
       #self.flattened_features = 128 * 3  
        self.flattened_features = 256 
        
        # LSTM για ακολουθίες
        self.lstm = nn.LSTM(
            input_size=self.flattened_features,  # Flattened features
            hidden_size=hidden_dim,             # Διαστάσεις hidden state
            num_layers=lstm_layers,             # Αριθμός LSTM layers
            batch_first=True,                   # Batch πρώτα
            dropout=dropout_rate,
        )

        # Fully Connected Layer για έξοδο
        self.fc = nn.Linear(hidden_dim, 1)  # Μία έξοδος ανά χρονικό βήμα

    def forward(self, x):
       """
       Είσοδος: x (B, T, C, F)
       """
   
       batch_size, seq_len, channels, features = x.size()
       ## print(f"Input size: {x.size()}")
        
        # Αναδιάταξη για CNN
       x = x.permute(0, 3, 1, 2)  # (B, F, T, C)
       # print(f"Permuted size for CNN: {x.size()}")
       
        # Πέρασμα από CNN
       x = self.cnn(x)  # Αναμένουμε 4D έξοδο
        #print(f"Output size after CNN: {x.size()}")
        
        # Flatten
       x = torch.flatten(x, start_dim=1)  # Flatten σε 2D
       # print(f"Flattened size: {x.size()}")
        
        # Αναδιάταξη για LSTM
       x = x.view(batch_size, seq_len, -1)  # (B, T, Flattened Features)
       # print(f"Reshaped size for LSTM: {x.size()}")
        
        # Πέρασμα από LSTM
       lstm_out, _ = self.lstm(x)
       # print(f"Output size after LSTM: {lstm_out.size()}")
        
        # Fully Connected Layer
       x = self.fc(lstm_out)
       # print(f"Output size after FC: {x.size()}")
        
       x = x.squeeze(-1)
       # print(f"Final output size: {x.size()}")
       return x

