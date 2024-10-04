import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# List of traffic movements
MOVEMENTS = ['EBL', 'EBR', 'EBT', 'NBL', 'NBR', 'NBT', 'SBL', 'SBR', 'SBT', 'WBL', 'WBR', 'WBT']

# Function to read TFT data
def read_TFT(int_num):
    dir_tft = "../result/KW_trained"
    data_time_idx = torch.load(os.path.join(dir_tft, f'TFTtensor_decoder_time_idx_{int_num}.pt'), map_location=torch.device('cpu'))
    data_target = torch.load(os.path.join(dir_tft, f'TFTtensor_decoder_target_{int_num}.pt'), map_location=torch.device('cpu'))
    data_prediction = torch.load(os.path.join(dir_tft, f'TFTtensor_prediction_{int_num}.pt'), map_location=torch.device('cpu'))
    
    time_idx = data_time_idx[0, :].cpu().numpy()
    Target = np.transpose(data_target[:, :].cpu().numpy())
    mean_Pred = data_prediction[:, :, [1, 3, 5]].cpu().numpy()
    mean_Pred = mean_Pred.transpose(1, 0, 2).reshape(96, 3 * 12)  # 0.1, 0.5, 0.9 quantiles
    
    return Target, mean_Pred

# Function to read TimesFM data
def read_TimesFM(int_num):
    data_prediction_TF = pd.read_csv(f'TimesFM_{int_num}_96_96.csv')
    data_prediction_TF = data_prediction_TF.pivot_table(index='Time_in', columns='Movement', values=['timesfm', 'timesfm-q-0.1', 'timesfm-q-0.9'], aggfunc='mean')
    data_prediction_TF = data_prediction_TF.swaplevel(axis=1).sort_index(axis=1, level='Movement')
    data_prediction_TF['time'] = pd.to_datetime(data_prediction_TF.index)
    return data_prediction_TF

# Function to reshape data for calculation
def dt_reshape(TimesFM_p, TFT_p, observed, movement_index):
    data_new = pd.DataFrame()
    data_new['time'] = TimesFM_p['time']
    TimesFM_p = TimesFM_p.to_numpy()
    move = movement_index * 3
    
    data_new['observed'] = pd.to_numeric(observed[:, movement_index], errors='coerce')
    data_new['TimesFM'] = pd.to_numeric(TimesFM_p[:, move], errors='coerce')
    data_new['TFT'] = pd.to_numeric(TFT_p[:, move + 1], errors='coerce')
    
    return data_new

# Function to calculate metrics per time step
def calculate_metrics_per_step(df, true_col, pred_col):
    y_true = df[true_col].values
    y_pred = df[pred_col].values
    
    # Ensure both are numeric
    y_true = pd.to_numeric(y_true, errors='coerce')
    y_pred = pd.to_numeric(y_pred, errors='coerce')
    
    # Calculate metrics at each time step
    mse = (y_true - y_pred) ** 2  # MSE at each time step
    mae = np.abs(y_true - y_pred)  # MAE at each time step
    mape = np.abs((y_true - y_pred) / (y_true + 0.5)) * 100  # MAPE
    
    return mse, mae, mape

# Function to plot comparison for TFT and TimesFM with smoother lines and better readability
def plot_metrics_comparison(metrics_TimesFM, metrics_TFT, time_steps, metric_name):
    plt.figure(figsize=(12, 6))
    
    # Apply rolling mean to smooth the lines
    metrics_TimesFM_smooth = pd.Series(metrics_TimesFM).rolling(window=5, min_periods=1).mean()
    metrics_TFT_smooth = pd.Series(metrics_TFT).rolling(window=5, min_periods=1).mean()
    
    # Plot with thicker lines for better visibility
    plt.plot(time_steps, metrics_TimesFM_smooth, label='TimesFM', color='blue', linewidth=2, linestyle='-')
    plt.plot(time_steps, metrics_TFT_smooth, label='TFT', color='orange', linewidth=2, linestyle='--')
    
    # Improve the title, labels, and grid
    plt.title(f'{metric_name} Comparison by Step (Summed Across Movements)', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add a legend
    plt.legend(fontsize=12)
    
    # Show the plot with tight layout to avoid label cutoff
    plt.tight_layout()
    plt.show()

# Main program that compares metrics per step across all movements
def main(intersection_numbers):
    """
    Main function to process multiple intersections and compare metrics per time step across all movements.
    
    Parameters:
        intersection_numbers (list): List of intersection numbers.
    """
    for int_num in intersection_numbers:
        observed, TFT_p = read_TFT(int_num)
        TimesFM_p = read_TimesFM(int_num)
        
        # Initialize variables to accumulate metrics across all movements
        total_mse_TimesFM = np.zeros(96)
        total_mae_TimesFM = np.zeros(96)
        total_mape_TimesFM = np.zeros(96)
        
        total_mse_TFT = np.zeros(96)
        total_mae_TFT = np.zeros(96)
        total_mape_TFT = np.zeros(96)
        
        for i in range(12):  # Iterate over all movements
            data_new = dt_reshape(TimesFM_p, TFT_p, observed, i)
            time_steps = data_new['time']
            
            # Calculate metrics per time step for both TimesFM and TFT
            mse_TimesFM, mae_TimesFM, mape_TimesFM = calculate_metrics_per_step(data_new, 'observed', 'TimesFM')
            mse_TFT, mae_TFT, mape_TFT = calculate_metrics_per_step(data_new, 'observed', 'TFT')
            
            # Accumulate metrics across all movements
            total_mse_TimesFM += mse_TimesFM
            total_mae_TimesFM += mae_TimesFM
            total_mape_TimesFM += mape_TimesFM
            
            total_mse_TFT += mse_TFT
            total_mae_TFT += mae_TFT
            total_mape_TFT += mape_TFT
        
        # Plotting the metrics comparison for each loss type (summed across all movements)
        plot_metrics_comparison(total_mse_TimesFM, total_mse_TFT, time_steps, 'MSE')
        plot_metrics_comparison(total_mae_TimesFM, total_mae_TFT, time_steps, 'MAE')
        plot_metrics_comparison(total_mape_TimesFM, total_mape_TFT, time_steps, 'MAPE')

# Example of calling the main function with a list of intersection numbers
if __name__ == "__main__":
    intersection_list = [0]  # Example intersection numbers
    main(intersection_list)
