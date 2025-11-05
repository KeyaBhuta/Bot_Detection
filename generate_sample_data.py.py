# Sample Data Generator for Botnet Detection
# Save this as: generate_sample_data.py
# Use this if you can't download the full dataset

import pandas as pd
import numpy as np
import random

print("=" * 60)
print("SAMPLE DATASET GENERATOR")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_normal_traffic(n_samples):
    """Generate normal network traffic patterns"""
    data = []
    
    for _ in range(n_samples):
        # Normal traffic characteristics
        dur = np.random.exponential(0.5)  # Short duration
        spkts = np.random.randint(5, 50)   # Moderate packets
        dpkts = int(spkts * np.random.uniform(0.8, 1.2))  # Similar response
        sbytes = spkts * np.random.randint(100, 1500)  # Reasonable bytes
        dbytes = dpkts * np.random.randint(100, 1500)
        rate = spkts / (dur + 0.001)
        
        # TCP characteristics
        sttl = 64
        dttl = 64
        sload = sbytes / (dur + 0.001)
        dload = dbytes / (dur + 0.001)
        sloss = 0
        dloss = 0
        sinpkt = dur / (spkts + 1)
        dinpkt = dur / (dpkts + 1)
        sjit = np.random.uniform(0, 0.01)
        djit = np.random.uniform(0, 0.01)
        swin = 255
        dwin = 255
        stcpb = random.randint(1000, 50000)
        dtcpb = random.randint(1000, 50000)
        tcprtt = np.random.uniform(0.01, 0.05)
        synack = np.random.uniform(0.005, 0.02)
        ackdat = np.random.uniform(0.005, 0.02)
        
        proto = 6  # TCP
        state = random.choice([0, 1, 2])  # FIN, CON, INT
        service = random.choice([0, 1, 2, 3])  # http, ftp, ssh, dns
        
        data.append([
            dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl,
            sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit,
            swin, dwin, stcpb, dtcpb, tcprtt, synack, ackdat,
            proto, state, service, 0  # Label: 0 = Normal
        ])
    
    return data

def generate_botnet_traffic(n_samples):
    """Generate botnet traffic patterns"""
    data = []
    
    for _ in range(n_samples):
        # Botnet characteristics: periodic beaconing, high volume
        dur = np.random.exponential(2.0)  # Longer duration
        spkts = np.random.randint(50, 500)  # Many packets
        dpkts = int(spkts * np.random.uniform(0.1, 0.3))  # Low response (C&C)
        sbytes = spkts * np.random.randint(50, 500)  # Small packets
        dbytes = dpkts * np.random.randint(50, 500)
        rate = spkts / (dur + 0.001)
        
        # Suspicious TCP characteristics
        sttl = random.choice([32, 64, 128, 254])  # Varied TTL
        dttl = random.choice([32, 64, 128, 254])
        sload = sbytes / (dur + 0.001)
        dload = dbytes / (dur + 0.001)
        sloss = np.random.randint(0, 10)  # Some packet loss
        dloss = np.random.randint(0, 5)
        sinpkt = np.random.uniform(0.001, 0.1)  # Regular intervals (beaconing)
        dinpkt = np.random.uniform(0.001, 0.1)
        sjit = np.random.uniform(0, 0.001)  # Low jitter (automated)
        djit = np.random.uniform(0, 0.001)
        swin = random.choice([8, 16, 32, 255])
        dwin = random.choice([8, 16, 32, 255])
        stcpb = random.randint(1000, 50000)
        dtcpb = random.randint(1000, 50000)
        tcprtt = np.random.uniform(0.05, 0.2)  # Higher RTT
        synack = np.random.uniform(0.02, 0.05)
        ackdat = np.random.uniform(0.02, 0.05)
        
        proto = random.choice([6, 17])  # TCP or UDP
        state = random.choice([1, 2, 3])  # Various states
        service = random.choice([4, 5, 6, 7])  # Unusual services
        
        data.append([
            dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl,
            sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit,
            swin, dwin, stcpb, dtcpb, tcprtt, synack, ackdat,
            proto, state, service, 1  # Label: 1 = Botnet
        ])
    
    return data

# Generate datasets
print("\n[STEP 1] Generating Training Dataset...")
n_normal_train = 3000
n_botnet_train = 3000

normal_data_train = generate_normal_traffic(n_normal_train)
botnet_data_train = generate_botnet_traffic(n_botnet_train)

# Combine and shuffle
all_data_train = normal_data_train + botnet_data_train
random.shuffle(all_data_train)

# Create DataFrame
columns = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'dwin', 'stcpb', 'dtcpb', 'tcprtt', 'synack', 'ackdat',
    'proto', 'state', 'service', 'label'
]

df_train = pd.DataFrame(all_data_train, columns=columns)

# Save training dataset
df_train.to_csv('UNSW_NB15_training-set.csv', index=False)
print(f"✓ Training dataset created: {df_train.shape}")
print(f"  - Normal traffic: {(df_train['label'] == 0).sum()}")
print(f"  - Botnet traffic: {(df_train['label'] == 1).sum()}")

# Generate test dataset
print("\n[STEP 2] Generating Test Dataset...")
n_normal_test = 100
n_botnet_test = 50

normal_data_test = generate_normal_traffic(n_normal_test)
botnet_data_test = generate_botnet_traffic(n_botnet_test)

all_data_test = normal_data_test + botnet_data_test
random.shuffle(all_data_test)

df_test = pd.DataFrame(all_data_test, columns=columns)
df_test.to_csv('test_data.csv', index=False)

print(f"✓ Test dataset created: {df_test.shape}")
print(f"  - Normal traffic: {(df_test['label'] == 0).sum()}")
print(f"  - Botnet traffic: {(df_test['label'] == 1).sum()}")

# Generate small demo dataset
print("\n[STEP 3] Generating Demo Dataset...")
n_normal_demo = 20
n_botnet_demo = 10

normal_data_demo = generate_normal_traffic(n_normal_demo)
botnet_data_demo = generate_botnet_traffic(n_botnet_demo)

all_data_demo = normal_data_demo + botnet_data_demo
random.shuffle(all_data_demo)

df_demo = pd.DataFrame(all_data_demo, columns=columns)
df_demo.to_csv('demo_data.csv', index=False)

print(f"✓ Demo dataset created: {df_demo.shape}")
print(f"  - Normal traffic: {(df_demo['label'] == 0).sum()}")
print(f"  - Botnet traffic: {(df_demo['label'] == 1).sum()}")

# Display sample data
print("\n[STEP 4] Sample Data Preview:")
print("\nFirst 5 rows:")
print(df_train.head())

print("\nDataset Statistics:")
print(df_train.describe())

print("\n" + "=" * 60)
print("SAMPLE DATA GENERATION COMPLETED!")
print("=" * 60)
print("\nGenerated Files:")
print("1. UNSW_NB15_training-set.csv (6000 flows) - For training")
print("2. test_data.csv (150 flows) - For testing Flask app")
print("3. demo_data.csv (30 flows) - For quick demonstration")
print("\nNext Steps:")
print("1. Run: jupyter notebook botnet_preprocessing.ipynb")
print("2. Run: python botnet_model_training.py")
print("3. Run: python app.py")
print("4. Upload test_data.csv or demo_data.csv to test the system")
print("=" * 60)