import os
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader
from datetime import datetime

# Ensure reproducibility
random.seed(0)
np.random.seed(0)

# Function to load configuration from a YAML file
def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

# Function to calculate and plot SHAP values
def calculate_and_plot_shap_values(model, dataloader, feature_names, cell_type_name, save_dir):
    """Calculate SHAP values and plot both summary and bar plots."""
    # Convert DataLoader dataset to a numpy array for SHAP compatibility
    data_list = []
    for data in dataloader:
        inputs, _ = data  # We only need the inputs
        data_list.append(inputs.numpy())

    data = np.vstack(data_list)  # Stack arrays vertically if your data loader uses batches

    # Initialize SHAP's GradientExplainer
    explainer = shap.GradientExplainer(model, torch.tensor(data).float())

    # Calculate SHAP values
    shap_values = explainer.shap_values(torch.tensor(data).float())

    # Plot the SHAP summary plot with the cell type in the title
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, data, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary Plot for {cell_type_name}")
    # Save the summary plot with the cell type name
    summary_plot_path = os.path.join(save_dir, f'{cell_type_name}_shap_summary_plot.png')
    plt.savefig(summary_plot_path)
    plt.close()

    # Plot the SHAP bar plot (global feature importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, data, feature_names=feature_names, plot_type="bar")
    plt.title(f"SHAP Feature Importance for {cell_type_name}")
    # Save the bar plot with the cell type name
    bar_plot_path = os.path.join(save_dir, f'{cell_type_name}_shap_feature_importance_plot.png')
    plt.savefig(bar_plot_path)
    plt.close()

    print(f"SHAP plots saved to {save_dir}")

# Main function to run the model and generate SHAP plots
def main(config_file):
    config = load_config(config_file)

    # Load datasets
    train = pd.read_csv(config['dataset']['train'], index_col=0)
    test = pd.read_csv(config['dataset']['test'], index_col=0)
    val = pd.read_csv(config['dataset']['val'], index_col=0)
    y_train = pd.read_csv(config['dataset']['y_train'])
    y_test = pd.read_csv(config['dataset']['y_test'])
    y_val = pd.read_csv(config['dataset']['y_val'])
    
    # Prepare data for training
    train_x = train.T
    test_x = test.T
    val_x = val.T
    train_y = y_train

    # Create datasets and dataloaders
    train_dataset = TabularDataset(train_x.T, train_y)
    test_dataset = TabularDataset(test_x.T, y_test)
    
    dataloader_params = {
        'batch_size': config['train']['batch_size'],
        'shuffle': False
    }
    test_dataloader = DataLoader(test_dataset, **dataloader_params)

    # Load the model (assuming it's saved from previous training)
    model_path = config['model_output']['model_path']
    model = torch.load(model_path)
    model.eval()

    # Generate and save SHAP plots
    feature_names = list(train_x.index)  # Assuming train_x is the dataframe with gene names as index
    cell_type_name = os.path.basename(config_file).replace("config_", "").replace(".yml", "")
    save_dir = config['model_output']['model_save_dir']
    calculate_and_plot_shap_values(model, test_dataloader, feature_names, cell_type_name, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SHAP analysis for a model')
    parser.add_argument('--configs', type=str, nargs='+', help='List of configuration files', required=True)
    args = parser.parse_args()
    
    for config_file in args.configs:
        print(f"Starting SHAP analysis with {config_file}")
        main(config_file)
        print(f"Finished SHAP analysis with {config_file}\n")


