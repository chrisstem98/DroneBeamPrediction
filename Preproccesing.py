# ============================================================
# STEP 2 - Data Preprocessing (70 / 10 / 20 split)
# ============================================================
# Reads scenario23_clean.csv produced by step1_load_data.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

# ============================================================
# 2.1 Load Clean CSV
# ============================================================
df = pd.read_csv('scenario23_clean.csv')

FEATURE_COLS = [
    'unit2_gps_lat',
    'unit2_gps_long',
    'pwr_max',
    'pwr_mean',
    'pwr_std',
    'pwr_median',
    'pwr_top3_mean',
    'pwr_range',
    'unit2_altitude',
    'unit2_distance',
    'unit2_speed',
    'unit2_height',
    'unit2_zspeed',
    'unit2_pitch'
]
LABEL_COL  = 'unit1_beam_index'
LABEL_TOP2 = 'unit1_beam_top2'
LABEL_TOP3 = 'unit1_beam_top3'

print(f"Samples loaded    : {len(df)}")
print(f"NaN values        :\n{df.isnull().sum()}")

df = df[FEATURE_COLS + [LABEL_COL, LABEL_TOP2, LABEL_TOP3]].dropna()
print(f"Samples after NaN removal : {len(df)}")

# ============================================================
# 2.2 Extract Features and Labels
# ============================================================
X = df[FEATURE_COLS].values
y      = df[LABEL_COL].values
y_top2 = df[LABEL_TOP2].values
y_top3 = df[LABEL_TOP3].values

print(f"\nX shape          : {X.shape}")
print(f"y shape          : {y.shape}")
print(f"Unique beams     : {len(np.unique(y))}")

# ============================================================
# 2.3 Normalization
# ============================================================
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 2.4 Train / Validation / Test Split (70 / 10 / 20)
# ============================================================
# Step 1: 80% train+val, 20% test
X_trainval, X_test, y_trainval, y_test, y2_trainval, y2_test, y3_trainval, y3_test = train_test_split(
    X_scaled, y, y_top2, y_top3,
    test_size=0.20,
    random_state=42
)

# Step 2: 12.5% of 80% = 10% of total for validation
X_train, X_val, y_train, y_val, y2_train, y2_val, y3_train, y3_val = train_test_split(
    X_trainval, y_trainval, y2_trainval, y3_trainval,
    test_size=0.125,
    random_state=42
)

total = len(X)
print(f"\n{'=' * 45}")
print("SPLIT RESULTS")
print(f"{'=' * 45}")
print(f"Total   : {total}")
print(f"Train   : {len(X_train)} ({len(X_train)/total*100:.1f}%)")
print(f"Val     : {len(X_val)}  ({len(X_val)/total*100:.1f}%)")
print(f"Test    : {len(X_test)} ({len(X_test)/total*100:.1f}%)")

# ============================================================
# 2.5 Save Preprocessed Data
# ============================================================
data = {
    'X_train'      : X_train,
    'X_val'        : X_val,
    'X_test'       : X_test,
    'y_train'      : y_train,
    'y_val'        : y_val,
    'y_test'       : y_test,
    'y2_train'     : y2_train,
    'y2_val'       : y2_val,
    'y2_test'      : y2_test,
    'y3_train'     : y3_train,
    'y3_val'       : y3_val,
    'y3_test'      : y3_test,
    'scaler'       : scaler,
    'feature_cols' : FEATURE_COLS,
    'df_original'  : df
}
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(data, f)

# ============================================================
# 2.6 Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(y_train, bins=64, alpha=0.6, label=f'Train ({len(X_train)})',    color='steelblue')
axes[0].hist(y_val,   bins=64, alpha=0.6, label=f'Val ({len(X_val)})',        color='orange')
axes[0].hist(y_test,  bins=64, alpha=0.6, label=f'Test ({len(X_test)})',      color='green')
axes[0].set_title('Beam Index Distribution per Split')
axes[0].set_xlabel('Beam Index')
axes[0].set_ylabel('Frequency')
axes[0].legend()

axes[1].bar(
    ['Train (70%)', 'Validation (10%)', 'Test (20%)'],
    [len(X_train), len(X_val), len(X_test)],
    color=['steelblue', 'orange', 'green']
)
axes[1].set_title('Split Sizes')
axes[1].set_ylabel('Number of Samples')
for i, v in enumerate([len(X_train), len(X_val), len(X_test)]):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('step2_preprocessing.png', dpi=150, bbox_inches='tight')
plt.show()
print("Step 2 complete. Saved: preprocessed_data.pkl, step2_preprocessing.png")