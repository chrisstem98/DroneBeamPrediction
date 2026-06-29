# ============================================================
# STEP 1 - Load and Explore Dataset (Scenario 23)
# ============================================================
# The scenario23.csv does not contain GPS values directly.
# It contains paths to separate files inside unit1/ and unit2/.
# This script reads those files and builds a single dataframe.
# Updated: added altitude, distance, speed, height, z-speed, pitch
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

lat_list          = []
lon_list          = []
beam_index_list   = []
beam_top2_list    = []
beam_top3_list    = []
max_pwr_list      = []
mean_pwr_list     = []
std_pwr_list      = []
median_pwr_list   = []
top3_mean_list    = []
pwr_range_list    = []
altitude_list     = []
distance_list     = []
speed_list        = []
height_list       = []
zspeed_list       = []
pitch_list        = []

print(f"\nReading {len(df_raw)} samples from external files...")

for i, row in df_raw.iterrows():
    try:
        # Read GPS file
        loc_path = os.path.join(ROOT, row['unit2_loc'].lstrip('./').replace('scenario23_dev/', ''))
        pos      = np.loadtxt(loc_path)
        lat_list.append(pos[0])
        lon_list.append(pos[1])

        # Read power file
        pwr_path = os.path.join(ROOT, row['unit1_pwr_60ghz'].lstrip('./').replace('scenario23_dev/', ''))
        pwr      = np.loadtxt(pwr_path)

        # Top-3 beam indices from actual power measurements
        top3_beams = np.argsort(pwr)[::-1][:3]
        beam_index_list.append(int(top3_beams[0]))
        beam_top2_list.append(int(top3_beams[1]))
        beam_top3_list.append(int(top3_beams[2]))

        # Power-based features
        top3_indices = np.argsort(pwr)[-3:]
        max_pwr_list.append(float(np.max(pwr)))
        mean_pwr_list.append(float(np.mean(pwr)))
        std_pwr_list.append(float(np.std(pwr)))
        median_pwr_list.append(float(np.median(pwr)))
        top3_mean_list.append(float(np.mean(pwr[top3_indices])))
        pwr_range_list.append(float(np.max(pwr) - np.min(pwr)))

# New features - read from external files like GPS
        speed_path    = os.path.join(ROOT, row['unit2_speed'].lstrip('./').replace('scenario23_dev/', ''))
        altitude_path = os.path.join(ROOT, row['unit2_altitude'].lstrip('./').replace('scenario23_dev/', ''))
        distance_path = os.path.join(ROOT, row['unit2_distance'].lstrip('./').replace('scenario23_dev/', ''))
        height_path   = os.path.join(ROOT, row['unit2_height'].lstrip('./').replace('scenario23_dev/', ''))
        zspeed_path   = os.path.join(ROOT, row['unit2_z-speed'].lstrip('./').replace('scenario23_dev/', ''))
        pitch_path    = os.path.join(ROOT, row['unit2_pitch'].lstrip('./').replace('scenario23_dev/', ''))

        speed_list.append(float(np.loadtxt(speed_path)))
        altitude_list.append(float(np.loadtxt(altitude_path)))
        distance_list.append(float(np.loadtxt(distance_path)))
        height_list.append(float(np.loadtxt(height_path)))
        zspeed_list.append(float(np.loadtxt(zspeed_path)))
        pitch_list.append(float(np.loadtxt(pitch_path)))

    except Exception as e:
        lat_list.append(np.nan)
        lon_list.append(np.nan)
        beam_index_list.append(np.nan)
        beam_top2_list.append(np.nan)
        beam_top3_list.append(np.nan)
        max_pwr_list.append(np.nan)
        mean_pwr_list.append(np.nan)
        std_pwr_list.append(np.nan)
        median_pwr_list.append(np.nan)
        top3_mean_list.append(np.nan)
        pwr_range_list.append(np.nan)
        altitude_list.append(np.nan)
        distance_list.append(np.nan)
        speed_list.append(np.nan)
        height_list.append(np.nan)
        zspeed_list.append(np.nan)
        pitch_list.append(np.nan)

    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{len(df_raw)} samples")

# ============================================================
# 1.3 Build Clean DataFrame
# ============================================================
df = pd.DataFrame({
    'unit2_gps_lat'    : lat_list,
    'unit2_gps_long'   : lon_list,
    'pwr_max'          : max_pwr_list,
    'pwr_mean'         : mean_pwr_list,
    'pwr_std'          : std_pwr_list,
    'pwr_median'       : median_pwr_list,
    'pwr_top3_mean'    : top3_mean_list,
    'pwr_range'        : pwr_range_list,
    'unit2_altitude'   : altitude_list,
    'unit2_distance'   : distance_list,
    'unit2_speed'      : speed_list,
    'unit2_height'     : height_list,
    'unit2_zspeed'     : zspeed_list,
    'unit2_pitch'      : pitch_list,
    'unit1_beam_index' : beam_index_list,
    'unit1_beam_top2'  : beam_top2_list,
    'unit1_beam_top3'  : beam_top3_list
})

df = df.dropna()
df['unit1_beam_index'] = df['unit1_beam_index'].astype(int)
df['unit1_beam_top2']  = df['unit1_beam_top2'].astype(int)
df['unit1_beam_top3']  = df['unit1_beam_top3'].astype(int)

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