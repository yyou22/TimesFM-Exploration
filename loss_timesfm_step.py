import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# List of traffic movements
MOVEMENTS = ['EBL', 'EBR', 'EBT', 'NBL', 'NBR', 'NBT', 'SBL', 'SBR', 'SBT', 'WBL', 'WBR', 'WBT']

# Function to read TimesFM data
def read_TimesFM(filename):
    data_prediction_TF = pd.read_csv(filename)
    data_prediction_TF = data_prediction_TF.pivot_table(
        index='Time_in', columns='Movement', values=['timesfm', 'timesfm-q-0.1', 'timesfm-q-0.9'], aggfunc='mean'
    )
    data_prediction_TF = data_prediction_TF.swaplevel(axis=1).sort_index(axis=1, level='Movement')
    data_prediction_TF['time'] = pd.to_datetime(data_prediction_TF.index)
    return data_prediction_TF

# Function to read observed data using TFT
def read_TFT(int_num):
    dir_tft = "../result/KW_trained"
    data_target = torch.load(os.path.join(dir_tft, f'TFTtensor_decoder_target_{int_num}.pt'), map_location=torch.device('cpu'))
    Target = np.transpose(data_target[:, :].cpu().numpy())
    return Target

# Function to reshape data for calculation
def dt_reshape(TimesFM_p, observed, movement_index):
    data_new = pd.DataFrame()
    data_new['time'] = TimesFM_p['time']
    TimesFM_p = TimesFM_p.to_numpy()
    move = movement_index * 3
    
    data_new['observed'] = pd.to_numeric(observed[:, movement_index], errors='coerce')
    data_new['TimesFM'] = pd.to_numeric(TimesFM_p[:, move], errors='coerce')
    
    return data_new

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 0.5))) * 100
    return mse, mae, mape

# Function to plot comparison for three TimesFM versions with smoother lines
def plot_metrics_comparison(metrics_list, labels, time_steps, metric_name):
    plt.figure(figsize=(12, 6))
    
    # Apply rolling mean to smooth the lines
    for metrics, label in zip(metrics_list, labels):
        smoothed_metrics = pd.Series(metrics).rolling(window=5, min_periods=1).mean()
        plt.plot(time_steps, smoothed_metrics, label=label, linewidth=2)
    
    plt.title(f'{metric_name} Comparison by Step (Summed Across Movements)', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# Main program to compare three TimesFM versions
def main(int_num, csv_files):
    observed = read_TFT(int_num)
    timesfm_data = [read_TimesFM(csv_file) for csv_file in csv_files]
    labels = ['TimesFM96', 'TimesFM192', 'TimesFM512']

    # Initialize metric accumulators
    total_metrics = {label: {'mse': np.zeros(96), 'rmse': np.zeros(96), 'mae': np.zeros(96), 'mape': np.zeros(96)}
                     for label in labels}

    weighted_metrics_sum = {label: np.zeros(3) for label in labels}  # Store weighted sums of MSE, MAE, MAPE
    total_volume_sum = 0

    for label, data in zip(labels, timesfm_data):
        for i in range(12):  # Iterate over all movements
            data_new = dt_reshape(data, observed, i)
            y_true = data_new['observed'].values
            y_pred = data_new['TimesFM'].values

            # Calculate metrics using Vincent's method
            mse, mae, mape = calculate_metrics(y_true, y_pred)
            volume = np.sum(y_true)
            weighted_metrics_sum[label] += np.array([mse, mae, mape]) * volume
            total_volume_sum += volume

            # Calculate metrics per time step for plotting
            total_metrics[label]['mse'] += (y_true - y_pred) ** 2
            total_metrics[label]['rmse'] += np.sqrt((y_true - y_pred) ** 2)
            total_metrics[label]['mae'] += np.abs(y_true - y_pred)
            total_metrics[label]['mape'] += np.abs((y_true - y_pred) / (y_true + 0.5)) * 100

    # Print overall weighted metrics using Vincent's method
    print("Overall Weighted Losses Across All Movements (Combined):")
    for label in labels:
        weighted_mse = weighted_metrics_sum[label][0] / total_volume_sum
        weighted_mae = weighted_metrics_sum[label][1] / total_volume_sum
        weighted_mape = weighted_metrics_sum[label][2] / total_volume_sum

        print(f"{label} - MSE: {weighted_mse:.4f}, MAE: {weighted_mae:.4f}, MAPE: {weighted_mape:.4f}")
    
    # Plotting the metrics comparison for each loss type (summed across all movements)
    time_steps = timesfm_data[0]['time']  # Use time steps from the first dataset for consistency
    metric_names = ['MSE', 'RMSE', 'MAE', 'MAPE']
    for metric_name in metric_names:
        plot_metrics_comparison(
            [total_metrics[label][metric_name.lower()] for label in labels],
            labels,
            time_steps,
            metric_name
        )

# Example of calling the main function with intersection number and CSV filenames
if __name__ == "__main__":
    intersection_num = 0
    csv_files = ['TimesFM_0_96_96.csv', 'TimesFM_0_192_96.csv', 'TimesFM_0_512_96.csv']  # Replace with actual CSV file paths
    main(intersection_num, csv_files)
