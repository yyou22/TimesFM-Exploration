import torch

# Function to load and print the content and size of a .pt file
def load_pt_file(file_path):
    try:
        # Load the .pt file
        data = torch.load(file_path, map_location=torch.device('cpu'))
        
        print("Contents of the .pt file:")
        
        # Check the type of data and print accordingly
        if isinstance(data, dict):
            print("Data type: Dictionary")
            for key, value in data.items():
                print(f"Key: {key}")
                
                # Check if the value has a size or shape attribute
                if hasattr(value, 'size') or hasattr(value, 'shape'):
                    print(f"Size/Shape: {value.size() if hasattr(value, 'size') else value.shape}")
                else:
                    print(f"Value: {value}")
                
                print("-" * 50)  # Separator between items
        elif isinstance(data, list):
            print("Data type: List")
            for idx, item in enumerate(data):
                print(f"Item {idx}:")
                
                # Check if the item has a size or shape attribute
                if hasattr(item, 'size') or hasattr(item, 'shape'):
                    print(f"Size/Shape: {item.size() if hasattr(item, 'size') else item.shape}")
                else:
                    print(f"Item: {item}")
                
                print("-" * 50)
        else:
            # Handle cases where the object is of another type
            print("Data type: Other")
            # Check if the data has size or shape
            if hasattr(data, 'size') or hasattr(data, 'shape'):
                print(f"Size/Shape: {data.size() if hasattr(data, 'size') else data.shape}")
            else:
                print(f"Value: {data}")
        
    except Exception as e:
        print(f"Error loading the .pt file: {e}")

# Provide the path to your .pt file
file_path = '../result/KW_trained/TFTtensor_prediction_0.pt'  # Replace with the actual path to your .pt file
#file_path = './My TFT/TFTtensor_prediction_0.pt' 

# Call the function to load and print the .pt file contents and size
load_pt_file(file_path)
