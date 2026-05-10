# ============================================================
# STEP 1 - Load and Explore Dataset (Scenario 23)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1.1 Load CSV
# ============================================================
# Place scenario23.csv in the same folder as this script
CSV_PATH = 'scenario23.csv'
df = pd.read_csv(CSV_PATH)

# ============================================================
# 1.2 Basic Exploration
# ============================================================
print("=" * 50)
print("DATASET BASIC INFO")
print("=" * 50)
print(f"Number of samples : {len(df)}")
print(f"Number of columns : {len(df.columns)}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nBasic statistics:\n{df.describe()}")
print(f"\nNaN values per column:\n{df.isnull().sum()}")

# ============================================================
# 1.3 Define Feature and Label Columns
# ============================================================
FEATURE_COLS = [
    'unit2_gps_lat',
    'unit2_gps_long',
    'unit2_altitude',
    'unit2_heading',
    'unit2_pitch',
    'unit2_yaw'
]
LABEL_COL = 'unit1_beam_index'

print(f"\nFeatures : {FEATURE_COLS}")
print(f"Label    : {LABEL_COL}")
print(f"\nBeam index distribution (top 10):\n{df[LABEL_COL].value_counts().head(10)}")

# ============================================================
# 1.4 Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc = axes[0].scatter(
    df['unit2_gps_long'],
    df['unit2_gps_lat'],
    c=df[LABEL_COL],
    cmap='viridis',
    s=5,
    alpha=0.7
)
plt.colorbar(sc, ax=axes[0], label='Beam Index')
axes[0].set_title('Drone Trajectory (colored by Beam Index)')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

axes[1].hist(df[LABEL_COL], bins=64, color='steelblue', edgecolor='white')
axes[1].set_title('Beam Index Distribution')
axes[1].set_xlabel('Beam Index')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('step1_exploration.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nStep 1 complete. Saved: step1_exploration.png")