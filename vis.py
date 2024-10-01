import pandas as pd
import matplotlib.pyplot as plt
import os
import re

dir = '../Dataset/dataset_KW/'
int_num = 0

index_csv = []
# Get the list of files and directories
index = os.listdir(dir)
# Filter out files that end with '.pkl'
index = [f for f in index if not f.endswith('.pkl')]
# Sort the list based on numeric values right before the .csv extension
index.sort(key=lambda f: int(re.search(r'(\d+)(?=\.csv)', f).group()) if re.search(r'(\d+)(?=\.csv)', f) else float('inf'))
index_csv = index

# Load actual data
local_ds = pd.read_csv(dir + index_csv[int_num])
local_ds = local_ds[local_ds['time_idx'] >= 44156]
# Select the first 96 rows
local_ds = local_ds.groupby('Movement').head(96)
local_ds = local_ds[['Time_in','Movement', 'Count']]
local_ds['Time_in'] = pd.to_datetime(local_ds['Time_in'],utc= True)
local_ds['Time_in'] = local_ds['Time_in'].dt.tz_convert('Canada/Eastern')
actual_ds = local_ds

#print(actual_ds['Movement'].unique())

# Load the forecasted data from the saved CSV file
forecast_ds = pd.read_csv('./TimesFM_' + str(int_num) + '_512_96' + '.csv')
forecast_ds['Time_in'] = pd.to_datetime(forecast_ds['Time_in'],utc= True)
forecast_ds['Time_in'] = forecast_ds['Time_in'].dt.tz_convert('Canada/Eastern')
#forecast_ds = forecast_ds[['Time_in', 'Movement', 'timesfm']]

#print(forecast_ds['Movement'].unique())

#print(actual_ds)
#print(forecast_ds)

# Merge the actual and forecasted data on Time_in and Movement
merged_df = pd.merge(actual_ds, forecast_ds[['Time_in', 'Movement', 'timesfm']], 
                     on=['Time_in', 'Movement'], suffixes=('_actual', '_forecast'))

# Plotting the actual vs forecasted values
plt.figure(figsize=(12, 6))

#print(merged_df['Movement'].unique())

# Select a specific 'Movement' for plotting, or loop through multiple movements
movement_id = 'SBR'  # Example movement, replace with desired one
subset = merged_df[merged_df['Movement'] == movement_id]

# Formatting the plot
plt.title(f"Actual vs Forecasted for Movement: {movement_id}")
plt.xlabel("Time")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Plot actual data
plt.plot(subset['Time_in'], subset['Count'], label='Actual', color='blue')

# Plot forecasted data
plt.plot(subset['Time_in'], subset['timesfm'], label='Forecasted', color='orange', linestyle='dashed')

# Show the plot
plt.tight_layout()
plt.show()
