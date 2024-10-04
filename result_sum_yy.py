# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:13:37 2024

@author: Vincent
"""

import os
import torch
import pandas as pd
import numpy as np
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
    #dir_tft = os.path.join("D:\\", "Desktop", "paper_4", "result", "MI_trained")

    dir_tft = "../result/KW_trained"
    
    data_time_idx = torch.load(os.path.join(dir_tft, f'TFTtensor_decoder_time_idx_{int_num}.pt'), map_location=torch.device('cpu'))

    data_target = torch.load(os.path.join(dir_tft, f'TFTtensor_decoder_target_{int_num}.pt'), map_location=torch.device('cpu'))
    data_prediction = torch.load(os.path.join(dir_tft, f'TFTtensor_prediction_{int_num}.pt'), map_location=torch.device('cpu'))
    
    time_idx = data_time_idx[0, :].cpu().numpy()  # Move to CPU for processing
    Target = np.transpose(data_target[:, :].cpu().numpy())
    mean_Pred = data_prediction[:, :, [1, 3, 5]].cpu().numpy()
    mean_Pred = mean_Pred.transpose(1, 0, 2).reshape(96, 3 * 12)  # 0.1, 0.5, 0.9 分位
    
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
    #data_prediction_TF = pd.read_csv(f'D:\\Desktop\\paper_4\\Result\\TimesFM_{int_num}.csv')

    data_prediction_TF = pd.read_csv('TimesFM_0_96_96.csv')
    data_prediction_TF = data_prediction_TF.pivot_table(index='Time_in', columns='Movement', values=['timesfm', 'timesfm-q-0.1', 'timesfm-q-0.9'], aggfunc='mean')
    data_prediction_TF = data_prediction_TF.swaplevel(axis=1).sort_index(axis=1, level='Movement')
    data_prediction_TF['time'] = pd.to_datetime(data_prediction_TF.index)  # 0.5, 0.1, 0.9 分位
    
    return data_prediction_TF

# Function to reshape data for calculation
def dt_reshape(TimesFM_p, TFT_p, observed, movement_index):
    """
    Reshapes data for calculation.
    
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

# Main program that loops through multiple intersections
def main(intersection_numbers):
    """
    Main function to process multiple intersections and store results in a DataFrame.
    
    Parameters:
        intersection_numbers (list): List of intersection numbers.
    
    Returns:
        pd.DataFrame: DataFrame containing the average metrics for each intersection,
                      and a row for the weighted metrics across all intersections.
    """
    results = []
    total_volume_sum = 0
    weighted_metrics_sum = np.zeros(10)  # To store weighted sum of all metrics (TimesFM + TFT)
    
    for int_num in intersection_numbers:
        observed, TFT_p = read_TFT(int_num)
        TimesFM_p = read_TimesFM(int_num)
        
        # Calculate and accumulate metrics for all movements in this intersection
        volume_sum = 0
        metrics_sum = np.zeros(5)
        metrics_TFT_sum = np.zeros(5)
        
        for i in range(12):  # Iterate over all movements
            data_new = dt_reshape(TimesFM_p, TFT_p, observed, i)
            
            volume = data_new['observed'].sum()
            volume_sum += volume
            
            metrics = np.array(calculate_metrics(data_new, 'observed', 'TimesFM'))
            metrics_TFT = np.array(calculate_metrics(data_new, 'observed', 'TFT'))
            
            metrics_sum += metrics * volume
            metrics_TFT_sum += metrics_TFT * volume
        
        # Calculate average metrics for this intersection
        average_metrics = metrics_sum / volume_sum
        average_metrics_TFT = metrics_TFT_sum / volume_sum
        
        # Store results in the list
        results.append([int_num, *average_metrics, *average_metrics_TFT, volume_sum])
        
        # Update the total volume sum and weighted sum of metrics
        total_volume_sum += volume_sum
        weighted_metrics_sum[:5] += average_metrics * volume_sum
        weighted_metrics_sum[5:] += average_metrics_TFT * volume_sum
    
    # Calculate weighted average metrics across all intersections
    weighted_average_metrics = weighted_metrics_sum / total_volume_sum
    
    # Convert results to a DataFrame
    columns = ['Intersection', 'RMSE_TimesFM', 'RMSPE_TimesFM', 'MAE_TimesFM', 'MAPE_TimesFM', 'R2_TimesFM',
               'RMSE_TFT', 'RMSPE_TFT', 'MAE_TFT', 'MAPE_TFT', 'R2_TFT', 'Volume_Sum']
    df_results = pd.DataFrame(results, columns=columns)
    
    # Add a final row for weighted metrics
    weighted_metrics_row = ['Weighted Average', *weighted_average_metrics, total_volume_sum]#解包操作
    df_results.loc[len(df_results)] = weighted_metrics_row
    
    return df_results

# Example of calling the main function with a list of intersection numbers
if __name__ == "__main__":
    #intersection_list = list(filter(lambda x: x != 72, range(61,76)))  # Example intersection numbers
    intersection_list = [0]
    result_df = main(intersection_list)
    print(result_df)
    result_df.to_csv("intersection_metrics.csv", index=False)