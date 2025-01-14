# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import joblib
import os
import pandas as pd
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import random_split

# Φόρτωση του αποθηκευμένου μοντέλου
model = joblib.load('best_model_SVR.joblib')

# Φόρτωση των δεδομένων
dfd = pd.read_csv('demographic.csv')
# Διαχείριση κατηγορικών δεδομένων (Sex με encoding)
dfd.loc[:, 'Sex'] = dfd['Sex'].map({'Male': 0, 'Female': 1})

# Επιλογή των χαρακτηριστικών που θα χρησιμοποιηθούν
df0 = dfd[['Sex', 'Age', 'Height']]

# Δημιουργία του αντικειμένου PolynomialFeatures με degree=2
poly = PolynomialFeatures(degree=1, interaction_only=True, include_bias=True, order='F')

# Εκπαίδευση του poly με τα δεδομένα (fit)
poly.fit(df0)

# Μετασχηματισμός των δεδομένων
X_test_poly = poly.transform(df0)

# Πρόβλεψη του βάρους με το αποθηκευμένο μοντέλο
dfd['Weight'] = model.predict(X_test_poly)

dfd.to_csv("/home/poulimenos/project/demographic.csv", index=False)

dfd = pd.read_csv('demographic.csv')
# Ορισμός φακέλων
root_folders = [
   Path("/home/poulimenos/project/output/koa/"),
   Path("/home/poulimenos/project/output/pd/"),
   Path("/home/poulimenos/project/output/nm/")
]



def compute_center_of_gravity(df):
    """
    Υπολογίζει το κέντρο βάρους (μέσος όρος των x, y, z) για τα επιλεγμένα σημεία 
    με βάση τις στήλες των συντεταγμένων για κάθε γραμμή.
    """
    # Επιλογή μόνο των στηλών που έχουν κατάληξη '_x', '_y', '_z'
    columns_with_x = [col for col in df.columns if col.endswith('_x')]
    columns_with_y = [col for col in df.columns if col.endswith('_y')]
    columns_with_z = [col for col in df.columns if col.endswith('_z')]
    
    # Υπολογισμός μέσου όρου των x, y, z για κάθε γραμμή
    df["CoMx"] = df[columns_with_x].mean(axis=1)  # Μέσος όρος των x συντεταγμένων για κάθε γραμμή
    df["CoMy"] = df[columns_with_y].mean(axis=1)  # Μέσος όρος των y συντεταγμένων για κάθε γραμμή
    df["CoM"] = df[columns_with_z].mean(axis=1)  # Μέσος όρος των z συντεταγμένων για κάθε γραμμή

    return df
def calculate_acceleration(df, delta_time=0.02):
    """
    Υπολογίζει την επιτάχυνση του άξονα y για τα σημεία με βάση το κέντρο βάρους.
    Υπολογίζει τη διαφορά θέσης και ταχύτητας με σταθερό χρονικό βήμα (50 fps).
    """
    # Υπολογισμός του κέντρου βάρους
    df = compute_center_of_gravity(df)
  
    # Διαφορά θέσης στον άξονα y
    df["delta_y"] = df["CoMy"].diff()
    
    # Υπολογισμός ταχύτητας
    df["velocity_y"] = df["delta_y"] / delta_time
   
    # Υπολογισμός επιτάχυνσης (διαφορά ταχύτητας προς το χρόνο)
    df["acceleration_y"] = df["velocity_y"].diff() / delta_time

    return df

def calculate_vgrf(df):
    """
    Υπολογίζει το VGRF (Vertical Ground Reaction Force) με βάση τις στήλες 'Weight' και 'acceleration_y'.
    Ελέγχει και διορθώνει μη αριθμητικά δεδομένα.
    
    Args:
        df (pd.DataFrame): Το DataFrame που περιέχει τις στήλες 'Weight' και 'acceleration_y'.

    Returns:
        pd.DataFrame: Το DataFrame με μια νέα στήλη 'VGRF'.
    """
    # Επιβεβαίωση ότι οι απαιτούμενες στήλες υπάρχουν
    required_columns = ['Weight', 'acceleration_y']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Η στήλη '{col}' δεν υπάρχει στο DataFrame.")
    
    
    # Σταθερά επιτάχυνσης λόγω βαρύτητας
    acceleration_of_gravity = 9.81  # m/s^2
    
    # Υπολογισμός VGRF
    df["VGRF"] = df["Weight"].str.replace(',', '.').astype(float) * (acceleration_of_gravity+ df["acceleration_y"])
    
    return df

# Αναζήτηση για .csv αρχεία και ενημέρωση των αντίστοιχων .csv αρχείων
for root_folder in root_folders:
    if not root_folder.exists():
        print(f"Folder not found: {root_folder}")
        continue  # Αν η διαδρομή δεν υπάρχει, παραλείπει τον φάκελο
    csv_files = list(root_folder.rglob("*.csv"))

    for csv_file in csv_files:
        print(f"Found CSV file: {csv_file}")
        filename = os.path.basename(csv_file)
        
        # Εύρεση ID στο όνομα αρχείου
        match = re.search(r"(\d{3})(\w+)_(\d{2})", filename)

        if match:
            # Ανάλυση του ονόματος του αρχείου για εξαγωγή πληροφοριών  +level gia koa,pd
            video_id, disease,side = match.groups()
            

        else:
            print(f"Invalid filename format for {filename}")
            continue  # Αγνόηση αρχείου αν δεν ταιριάζει το πρότυπο
        
        # Φόρτωση δεδομένων από .csv
        try:
            data = pd.read_csv(csv_file)
            print(f"Loaded data from {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

        # Έλεγχος αν η στήλη `Disease` υπάρχει στο DataFrame
        if 'Disease' not in data.columns:
            print(f"Column 'Disease' not found in {filename}")
            continue
        data['LEFT_CLOSED_TO_CAMERA'] = (data['Side'] == 2).astype(int)
        data['RIGHT_CLOSED_TO_CAMERA'] = (data['Side'] == 1).astype(int)
        if data["Disease"][1]== "NM":
            data['Id'] =(data["ID"].astype(str)+data["Disease"])
        else:
           data['Id'] =(data["ID"].astype(str)+data["Disease"]+'_'+data["Level"])
        print(data['Id'])
        data=pd.merge(data, dfd, left_on='Id', right_on='VIDEO_CODE', how='inner') 
        data= calculate_acceleration(data)
        data['acceleration_y'] = data['acceleration_y'].fillna(0)
        data=calculate_vgrf(data)
        # Ενημέρωση του ίδιου αρχείου CSV με τις τροποποιήσεις
        try:

           output_folder = Path("/home/poulimenos/project/my_data/")
            

           # Καθορισμός του αρχείου εξόδου
           output_csv = output_folder / f"set_{video_id}{disease}_{side}.csv"

          # Αποθήκευση των δεδομένων

           data.to_csv(output_csv, index=False)
           print(f"Processed {filename}, results saved to {output_csv}")
        except Exception as e:
           print(f"Error saving updated file {csv_file}: {e}")




