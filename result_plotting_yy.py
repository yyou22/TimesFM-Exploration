# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:13:37 2024

@author: Vincent
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# List of traffic movements
MOVEMENTS = ['EBL', 'EBR', 'EBT', 'NBL', 'NBR', 'NBT', 'SBL', 'SBR', 'SBT', 'WBL', 'WBR', 'WBT']

# Function to read TFT data
def read_TFT(int_num):
    """
    Reads TFT data from the specified directory.
    
    Parameters:
        int_num (int): Intersection number.
    
    Returns:
        Target (np.ndarray): Observed values.
        mean_Pred (np.ndarray): Predicted values.
    """
    #dir_tft = os.path.join("D:\\", "Desktop", "paper_4", "Result", "TFT")
    dir_tft = "../result/KW_trained"
    
    data_time_idx = torch.load(os.path.join(dir_tft, f'TFTtensor_decoder_time_idx_{int_num}.pt'), map_location=torch.device('cpu'))
    data_target = torch.load(os.path.join(dir_tft, f'TFTtensor_decoder_target_{int_num}.pt'), map_location=torch.device('cpu'))
    data_prediction = torch.load(os.path.join(dir_tft, f'TFTtensor_prediction_{int_num}.pt'), map_location=torch.device('cpu'))
    
    time_idx = data_time_idx[0, :].cpu().numpy()  # Move to CPU for processing
    Target = np.transpose(data_target[:, :].cpu().numpy())
    mean_Pred = data_prediction[:, :, [1, 3, 5]].cpu().numpy()
    mean_Pred = mean_Pred.transpose(1, 0, 2).reshape(96, 3 * 12) #0.1,0.5,0.9分位
    
    return Target, mean_Pred

# Function to read TimesFM data
def read_TimesFM(int_num):
    """
    Reads TimesFM prediction data.
    
    Parameters:
        int_num (int): Intersection number.
    
    Returns:
        pd.DataFrame: Processed TimesFM data.
    """
    data_prediction_TF = pd.read_csv('TimesFM_0_96_96.csv')
    data_prediction_TF = data_prediction_TF.pivot_table(index='Time_in', columns='Movement', values=['timesfm', 'timesfm-q-0.1','timesfm-q-0.9'], aggfunc='mean')
    data_prediction_TF = data_prediction_TF.swaplevel(axis=1).sort_index(axis=1, level='Movement')
    data_prediction_TF['time'] = pd.to_datetime(data_prediction_TF.index)#0.5,0.1,0.9分位
    
    return data_prediction_TF

# Function to reshape data for plotting
def dt_reshape(TimesFM_p, TFT_p, observed, movement_index):
    """
    Reshapes data for plotting.
    
    Parameters:
        TimesFM_p (pd.DataFrame): TimesFM prediction data.
        TFT_p (np.ndarray): TFT prediction data.
        observed (np.ndarray): Observed values.
        movement_index (int): Index of the traffic movement.
    
    Returns:
        pd.DataFrame: Reshaped data.
    """
    data_new = pd.DataFrame()
    data_new['time'] = TimesFM_p['time']
    TimesFM_p = TimesFM_p.to_numpy()
    move = movement_index * 3
    
    data_new['observed'] = observed[:, movement_index]
    data_new['TimesFM'] = TimesFM_p[:, move]
    data_new['TimesFM_U'] = pd.to_numeric(TimesFM_p[:, move + 2], errors='coerce')
    data_new['TimesFM_L'] = pd.to_numeric(TimesFM_p[:, move + 1], errors='coerce')
    data_new['TFT'] = TFT_p[:, move + 1]
    data_new['TFT_U'] = TFT_p[:, move + 2]
    data_new['TFT_L'] = TFT_p[:, move]
    
    return data_new

# Function to plot data
def plotting(time_index, observed, prediction, prediction_u, prediction_b, y_label):
    """
    Plots the observed and predicted traffic volumes.
    
    Parameters:
        time_index (pd.Series): Time index for x-axis.
        observed (np.ndarray): Observed values.
        prediction (np.ndarray): Predicted values.
        prediction_u (np.ndarray): Upper bound predictions.
        prediction_b (np.ndarray): Lower bound predictions.
        y_label (str): Y-axis label.
    """
    plt.figure(figsize=(16, 10), dpi=100)
    plt.ylabel(f"{y_label} (vehs/hour)")
    plt.xlabel("Time of Day (hour)")
    mpl.rcParams['font.size'] = 30
    time_formatter = mdates.DateFormatter("%H")
    plt.gca().xaxis.set_major_formatter(time_formatter)
    plt.grid(True)
    
    plt.plot(time_index, observed, color="black", lw=3, label="Observations")
    plt.plot(time_index, prediction, color="#008080", linestyle="--", lw=3, label="Predictions") 
    plt.fill_between(time_index, prediction_u, prediction_b, color="#ADD8E6", alpha=0.4)
    plt.gca().spines["top"].set_alpha(1)
    plt.gca().spines["bottom"].set_alpha(1)
    plt.gca().spines["right"].set_alpha(1)
    plt.gca().spines["left"].set_alpha(1)
    plt.legend()
    plt.show()

# Function to calculate metrics
def calculate_metrics(df, true_col, pred_col):
    """
    Calculates error metrics for predictions.
    
    Parameters:
        df (pd.DataFrame): Data containing observed and predicted values.
        true_col (str): Column name for true values.
        pred_col (str): Column name for predicted values.
    
    Returns:
        tuple: RMSE, RMSPE, MAE, MAPE, R²
    """
    y_true = df[true_col].values
    y_pred = df[pred_col].values
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate RMSPE
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + 0.5)))) * 100
    
    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 0.5))) * 100
    
    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    
    return rmse, rmspe, mae, mape, r2

# Main program
int_num = 0# Intersection number
observed, TFT_p = read_TFT(int_num)
TimesFM_p = read_TimesFM(int_num)



# Calculate and accumulate metrics for all movements
volume_sum = 0
metrics_sum = np.zeros(5)
metrics_TFT_sum = np.zeros(5)

for i in range(12):  # Iterate over all movements
    data_new = dt_reshape(TimesFM_p, TFT_p, observed, i)
    plotting(data_new['time'], data_new['observed'], data_new['TimesFM'], data_new['TimesFM_U'], data_new['TimesFM_L'], MOVEMENTS[i])
    plotting(data_new['time'], data_new['observed'], data_new['TFT'], data_new['TFT_U'], data_new['TFT_L'], MOVEMENTS[i])
    
    volume = data_new['observed'].sum()
    volume_sum += volume
    
    metrics = np.array(calculate_metrics(data_new, 'observed', 'TimesFM'))
    metrics_TFT = np.array(calculate_metrics(data_new, 'observed', 'TFT'))
    
    metrics_sum += metrics*volume
    metrics_TFT_sum += metrics_TFT*volume

# Calculate average metrics
average_metrics = metrics_sum / volume_sum
average_metrics_TFT = metrics_TFT_sum / volume_sum

print(f"Average Metrics (TimesFM): RMSE={average_metrics[0]:.4f}, RMSPE={average_metrics[1]:.4f}%, MAE={average_metrics[2]:.4f}, MAPE={average_metrics[3]:.4f}%, R²={average_metrics[4]:.4f}")
print(f"Average Metrics (TFT): RMSE={average_metrics_TFT[0]:.4f}, RMSPE={average_metrics_TFT[1]:.4f}%, MAE={average_metrics_TFT[2]:.4f}, MAPE={average_metrics_TFT[3]:.4f}%, R²={average_metrics_TFT[4]:.4f}")
