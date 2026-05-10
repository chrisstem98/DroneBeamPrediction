# ============================================================
# STEP 1 - Load and Explore Dataset (Scenario 23)
# ============================================================
# The scenario23.csv does not contain GPS values directly.
# It contains paths to separate files inside unit1/ and unit2/.
# This script reads those files and builds a single dataframe.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1.1 Set Root Folder
# ============================================================
# Set this to the folder where you extracted the zip file.
# Example: ROOT = 'C:/Users/yourname/Downloads/scenario23_dev'
ROOT     = 'uav_beam_project/scenario23_dev'
CSV_PATH = os.path.join(ROOT, 'scenario23.csv')

df_raw = pd.read_csv(CSV_PATH)

print("=" * 50)
print("RAW CSV INFO")
print("=" * 50)
print(f"Number of rows    : {len(df_raw)}")
print(f"Columns           : {df_raw.columns.tolist()}")
print(f"\nFirst row:\n{df_raw.iloc[0]}")

# ============================================================
# 1.2 Read GPS and Beam Data from External Files
# ============================================================
# unit2_loc  -> txt file with [latitude, longitude]
# unit1_pwr_60ghz -> txt file with power per beam (64 values)
# unit1_beam_index -> index of max power = optimal beam

lat_list        = []
lon_list        = []
beam_index_list = []

print(f"\nReading {len(df_raw)} samples from external files...")

for i, row in df_raw.iterrows():
    try:
        # Read GPS file
        loc_path = os.path.join(ROOT, row['unit2_loc'].lstrip('./').replace('scenario23_dev/', ''))
        pos      = np.loadtxt(loc_path)
        lat_list.append(pos[0])
        lon_list.append(pos[1])

        # Read power file and compute beam index
        pwr_path  = os.path.join(ROOT, row['unit1_pwr_60ghz'].lstrip('./').replace('scenario23_dev/', ''))
        pwr       = np.loadtxt(pwr_path)
        beam_idx  = int(np.argmax(pwr))
        beam_index_list.append(beam_idx)

    except Exception as e:
        lat_list.append(np.nan)
        lon_list.append(np.nan)
        beam_index_list.append(np.nan)

    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{len(df_raw)} samples")

# ============================================================
# 1.3 Build Clean DataFrame
# ============================================================
df = pd.DataFrame({
    'unit2_gps_lat'    : lat_list,
    'unit2_gps_long'   : lon_list,
    'unit1_beam_index' : beam_index_list
})

df = df.dropna()
df['unit1_beam_index'] = df['unit1_beam_index'].astype(int)

print(f"\nClean dataset size: {len(df)} samples")
print(f"\nSample rows:\n{df.head(5)}")
print(f"\nBeam index range: {df['unit1_beam_index'].min()} to {df['unit1_beam_index'].max()}")
print(f"Unique beam indices: {df['unit1_beam_index'].nunique()}")

# ============================================================
# 1.4 Save Clean CSV for Next Steps
# ============================================================
df.to_csv('scenario23_clean.csv', index=False)
print("\nSaved clean dataset to: scenario23_clean.csv")

# ============================================================
# 1.5 Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc = axes[0].scatter(
    df['unit2_gps_long'],
    df['unit2_gps_lat'],
    c=df['unit1_beam_index'],
    cmap='viridis',
    s=5,
    alpha=0.7
)
plt.colorbar(sc, ax=axes[0], label='Beam Index')
axes[0].set_title('Drone Trajectory (colored by Beam Index)')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

axes[1].hist(df['unit1_beam_index'], bins=64, color='steelblue', edgecolor='white')
axes[1].set_title('Beam Index Distribution')
axes[1].set_xlabel('Beam Index')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('step1_exploration.png', dpi=150, bbox_inches='tight')
plt.show()
print("Step 1 complete. Saved: scenario23_clean.csv, step1_exploration.png")