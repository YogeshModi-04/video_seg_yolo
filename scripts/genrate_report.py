import os
import pandas as pd

# Define the directory containing the model reports
root_dir = './report/'  # Adjust this path if necessary
output_file = 'combined_report.csv'

# Initialize an empty DataFrame to hold all data
all_data = pd.DataFrame()

# Walk through each test folder in the directory
for test_folder in os.listdir(root_dir):
    test_folder_path = os.path.join(root_dir, test_folder)
    
    # Check if it's a directory
    if os.path.isdir(test_folder_path):
        # Now look inside each model format folder (onnx, TensorRT, Pt)
        for model_format in os.listdir(test_folder_path):
            model_format_path = os.path.join(test_folder_path, model_format)
            
            # Check if it's a directory
            if os.path.isdir(model_format_path):
                
                # Iterate over each CSV file within the model format folder
                for csv_file in os.listdir(model_format_path):
                    if csv_file.endswith('.csv'):
                        
                        # Extract model name from the file name (without extension)
                        model_name = os.path.splitext(csv_file)[0]
                        
                        # Load the CSV data
                        csv_path = os.path.join(model_format_path, csv_file)
                        
                        try:
                            data = pd.read_csv(csv_path)
                            
                            # Add columns for test folder, model format, and model name
                            data['Test_Folder'] = test_folder
                            data['Model_Format'] = model_format
                            data['Model_Name'] = model_name
                            
                            # Append the data to the all_data DataFrame
                            all_data = pd.concat([all_data, data], ignore_index=True)
                        except Exception as e:
                            print(f"Error reading {csv_file}: {e}")

# Restructure the data into a table format
restructured_report = all_data.pivot_table(
    index=['Test_Folder', 'Model_Format', 'Model_Name'],
    columns='Metric',
    values='Value',
    aggfunc='first'  # Use first as there's only one value per metric in each file
).reset_index()

# Flatten the columns
restructured_report.columns.name = None  # Remove the name of the column index
restructured_report.columns = [str(col) for col in restructured_report.columns]  # Flatten

# Save the restructured data to a CSV file
restructured_output_file = 'report.csv'
restructured_report.to_csv(restructured_output_file, index=False)
print(f"Restructured combined report saved to {restructured_output_file}")

# Display the first few rows to verify
print(restructured_report.head())
