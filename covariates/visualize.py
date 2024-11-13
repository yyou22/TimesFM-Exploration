import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
csv_file = 'forecast_results_with_movements.csv'  # Replace with your actual file path
output_folder = 'forecast_visualizations'
os.makedirs(output_folder, exist_ok=True)

# Read the CSV data
data = pd.read_csv(csv_file)

# Get a list of unique traffic movements
unique_movements = data['unique_id'].unique()

# Iterate through each movement and create a plot
for movement in unique_movements:
    # Filter data for the specific movement
    movement_data = data[data['unique_id'] == movement]
    
    # Extract forecasted and actual values
    forecast_values = movement_data.filter(like='forecast').values.flatten()
    actual_values = movement_data.filter(like='actual').values.flatten()
    
    # Plot the forecast and actual values
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_values, label='Forecast', color='blue')
    plt.plot(actual_values, label='Actual', color='orange')
    
    # Add titles and labels
    plt.title(f'Traffic Movement: {movement}')
    plt.xlabel('Time Step')
    plt.ylabel('Count')
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(output_folder, f'forecast_vs_actual_{movement}.png'))
    plt.close()

print(f"Visualizations saved in the '{output_folder}' folder.")
