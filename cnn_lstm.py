#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 22:13:27 2024

@author: poulimenos
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv
from cnn_lstm_class import CNNLSTM 
from csv_dataset_class import CustomCSVData



# Συνάρτηση για την καταμέτρηση αρχείων CSV σε ένα φάκελο
def count_csv_files(folder_path):
    """
    Μετράει τα αρχεία CSV σε έναν φάκελο.

    Parameters:
    - folder_path (str): Διαδρομή προς το φάκελο.

    Returns:
    - int: Αριθμός αρχείων CSV στο φάκελο.
    """
    if not os.path.isdir(folder_path):  # Έλεγχος αν η διαδρομή είναι φάκελος
        raise ValueError(f"{folder_path} is not a valid directory.")

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]  # Φιλτράρισμα μόνο αρχείων CSV
    return len(csv_files)  # Επιστροφή του πλήθους των αρχείων

# Συνάρτηση για καταμέτρηση γραμμών σε ένα αρχείο CSV
def count_line_csv(s):
    with open(s, "r") as file:  # Άνοιγμα του αρχείου σε λειτουργία ανάγνωσης
        reader = csv.reader(file)
        n = sum(1 for row in reader) - 1  # Υπολογισμός των γραμμών, αφαιρώντας τη γραμμή κεφαλίδας
    return n

# Συνάρτηση εκπαίδευσης για ένα υποκείμενο
def train_one_subject(model, train_file_path, epoch, num_subject, seq_len, batch_size, learning_rate, device, l1_lambda, l2_lambda):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)  # Χρήση του Adam optimizer με L2 regularization
    train_dataset = CustomCSVData([train_file_path], sequence_length=seq_len)  # Δημιουργία dataset από το αρχείο CSV

    n = count_line_csv(train_file_path) // (seq_len * batch_size)  # Υπολογισμός batch
    train_losses = []

    for i in range(n):  # Επανάληψη για κάθε batch
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Δημιουργία DataLoader
        model.train()  # Το μοντέλο τίθεται σε λειτουργία εκπαίδευσης
        train_loss = 0

        for features, labels in train_loader:  # Επανάληψη στα δεδομένα του batch
            features, labels = features.to(device), labels.to(device)  # Μεταφορά στο σωστό device (π.χ. GPU)
            optimizer.zero_grad()  # Μηδενισμός των gradients
            outputs = model(features)  # Υπολογισμός εξόδων από το μοντέλο
            loss = nn.MSELoss()(outputs.squeeze(), labels)  # Υπολογισμός της βασικής απώλειας (MSE)

            # Προσθήκη L1 και L2 κανονικοποίησης στην απώλεια
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss += l1_lambda * l1_norm + l2_lambda * l2_norm

            loss.backward()  # Υπολογισμός των gradients
            optimizer.step()  # Ενημέρωση των παραμέτρων
            train_loss += loss.item()  # Προσθήκη απώλειας

        train_loss /= len(train_loader)  # Μέση απώλεια ανά batch
        print(f"Train Loss {i}: {train_loss:.4f}")  # Εκτύπωση της απώλειας για κάθε batch
        train_losses.append(train_loss)

    avg_train_losses = sum(train_losses) / len(train_losses)  # Μέση απώλεια σε όλα τα batches
    print(f"End with this subject {train_file_path}, AVG Loss: {avg_train_losses}")  # Εκτύπωση μέσης απώλειας
    return avg_train_losses, model  # Επιστροφή των αποτελεσμάτων

# Συνάρτηση για επικύρωση του μοντέλου
def validate_model(model, val_data_paths, seq_len, batch_size, device):
    val_files = [os.path.join(val_data_paths, f) for f in os.listdir(val_data_paths) if f.endswith('.csv')]  # Εύρεση αρχείων επικύρωσης
    model.eval()  # Θέτουμε το μοντέλο σε λειτουργία αξιολόγησης
    total_val_loss = 0
    val_batch_count = 0
    criterion = nn.MSELoss()  # Ορισμός της συνάρτησης απώλειας

    with torch.no_grad():  # Απενεργοποίηση της καταγραφής gradients
        for val_file in val_files:
            val_dataset = CustomCSVData([val_file], sequence_length=seq_len)  # Δημιουργία dataset επικύρωσης
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Δημιουργία DataLoader
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)
                total_val_loss += loss.item()  # Συνολική απώλεια επικύρωσης
                val_batch_count += 1

    avg_val_loss = total_val_loss / max(1, val_batch_count)  # Μέση απώλεια επικύρωσης
    return avg_val_loss
def train_and_validate(model, train_data_paths, val_data_paths, learning_rate=0.001, batch_size=16, patience=5, device='cpu', num_subject=10, seq_len=20, l1_lambda=0.0001, l2_lambda=0.0001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Ρύθμιση του optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Scheduler για μείωση του learning rate

    model.to(device)  # Μεταφορά του μοντέλου στη συσκευή (CPU ή GPU)
    
    train_losses = []  # Αποθήκευση των απωλειών εκπαίδευσης
    val_losses = []  # Αποθήκευση των απωλειών επικύρωσης

    epoch = 0  # Μετρητής για τα υποκείμενα που επεξεργαζόμαστε
    l = os.listdir(train_data_paths)  # Λίστα με τα αρχεία στην τοποθεσία εκπαίδευσης
    # Εκπαίδευση για κάθε υποκείμενο
    for f in l:
        file_path = os.path.join(train_data_paths, f)  # Δημιουργία του πλήρους μονοπατιού του αρχείου
        if file_path.endswith('.csv'):  # Έλεγχος αν το αρχείο είναι CSV
            print(f"Epoch {epoch+1}/{num_subject} subject {f}")  # Εμφάνιση προόδου
            
            # Εκπαίδευση για ένα υποκείμενο
            train_loss, model = train_one_subject(model, file_path, epoch, num_subject, seq_len, batch_size, learning_rate, device, l1_lambda=l1_lambda, l2_lambda=l2_lambda)
            train_losses.append(train_loss)  # Προσθήκη της απώλειας εκπαίδευσης στη λίστα
            
            # Επικύρωση για κάθε υποκείμενο
            val_loss = validate_model(model, val_data_paths, seq_len, batch_size, device)
            val_losses.append(val_loss)  # Προσθήκη της απώλειας επικύρωσης στη λίστα
            
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")  # Εμφάνιση απωλειών
            epoch += 1  # Αύξηση του μετρητή epochs
         
            optimizer.step()  # Ενημέρωση των παραμέτρων του μοντέλου
    print("End with this group")  # Εμφάνιση μηνύματος τέλους
    # Εμφάνιση γραφημάτων για τις απώλειες
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()

    return model  # Επιστροφή του εκπαιδευμένου μοντέλου

#=====================================================
#=====================================================
# Εκτέλεση του script
if __name__ == "__main__":
    # Ορισμός διαδρομών προς τα datasets
    train_data_paths_koa = '/home/poulimenos/project/my_data/side_1/train/koa'
    val_data_paths_koa = '/home/poulimenos/project/my_data/side_1/valid/koa'
    train_data_paths_nm = '/home/poulimenos/project/my_data/side_1/train/nm'
    val_data_paths_nm = '/home/poulimenos/project/my_data/side_1/valid/nm'
    train_data_paths_pd = '/home/poulimenos/project/my_data/side_1/train/pd'
    val_data_paths_pd = '/home/poulimenos/project/my_data/side_1/valid/pd'
    
    # Καταμέτρηση αρχείων CSV σε κάθε διαδρομή
    train_koa_count = count_csv_files(train_data_paths_koa)
    val_koa_count = count_csv_files(val_data_paths_koa)
    train_nm_count = count_csv_files(train_data_paths_nm)
    val_nm_count = count_csv_files(val_data_paths_nm)
    train_pd_count = count_csv_files(train_data_paths_pd)
    val_pd_count = count_csv_files(val_data_paths_pd)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Ορισμός της συσκευής

    print("Initializing model...")  # Έναρξη μοντέλου
    
    model = CNNLSTM(hidden_dim=128, lstm_layers=2, dropout_rate=0.5)  # Ορισμός του μοντέλου
    model.to(device)  # Μεταφορά του μοντέλου στη συσκευή
    
    # Εκπαίδευση στο σύνολο δεδομένων "koa"
    print("Training on Koa dataset")
    model = train_and_validate(model, train_data_paths_koa, val_data_paths_koa, learning_rate=0.001, device=device, num_subject=train_koa_count)

    # Εκπαίδευση στο σύνολο δεδομένων "nm"
    print("Training on NM dataset")
    model = train_and_validate(model, train_data_paths_nm, val_data_paths_nm, learning_rate=0.001, device=device, num_subject=train_nm_count)

    # Εκπαίδευση στο σύνολο δεδομένων "pd"
    print("Training on Pd dataset")
    model = train_and_validate(model, train_data_paths_pd, val_data_paths_pd, learning_rate=0.001, device=device, num_subject=train_pd_count)

    test_losses = []  # Λίστα για τις απώλειες του test set
    
    # Ορισμός optimizer, loss function και scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if os.path.isdir('/home/poulimenos/project/my_data/side_1/test'):  # Έλεγχος αν η διαδρομή είναι φάκελος
        l = os.listdir('/home/poulimenos/project/my_data/side_1/test')  # Λίστα με τα αρχεία
        for f in l:
            file_path = os.path.join('/home/poulimenos/project/my_data/side_1/test', f)
            if file_path.endswith('.csv'):  # Έλεγχος αν το αρχείο είναι CSV
                test_dataset = CustomCSVData([file_path], sequence_length=20)  # Δημιουργία dataset από το αρχείο
                test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # Δημιουργία DataLoader
                
                model.train()  # Ορισμός του μοντέλου σε κατάσταση εκπαίδευσης
                test_loss = 0

                # Βρόχος εκπαίδευσης
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)  # Μεταφορά δεδομένων στη συσκευή
                    optimizer.zero_grad()  # Μηδενισμός των gradients
                    outputs = model(features)  # Προβλέψεις του μοντέλου
                    loss = criterion(outputs.squeeze(), labels)  # Υπολογισμός της απώλειας
                    loss.backward()  # Υπολογισμός των gradients
                    optimizer.step()  # Ενημέρωση των παραμέτρων
                    test_loss += loss.item()  # Προσθήκη της απώλειας

                test_loss /= len(test_loader)   # Μέση απώλεια
                print(f"Test Loss :{test_loss}")
                test_losses.append(test_loss)  # Προσθήκη της απώλειας στη λίστα
    
    avg_test_loss = sum(test_losses) / len(test_losses)  # Υπολογισμός μέσης απώλειας
    print(f"Average Test Loss: {avg_test_loss}")  # Εμφάνιση μέσης απώλειας