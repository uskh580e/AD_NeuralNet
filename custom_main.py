import os
import torch
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import argparse
import random  # Import random module
import shap  # Import SHAP
import matplotlib.pyplot as plt  # For visualizations
import pandas as pd
import yaml
import copy
from datetime import datetime
import csv
from utils import *
from gene_expression import *
from pathway_hierarchy import *
from custom_neural_network import *
from custom_fc_network import *

random.seed(0)
np.random.seed(0)

model_dct = dict()
# Deep Blue for the bar plot
bar_color = '#2ecc71'  # Sky Blue for the bar plot
summary_color = '#f39c12'  # Golden Yellow for the summary plot

# Hook function
def hook_fn(module, input, output, layer_name):
    global model_dct
    print(f"Hook called for layer: {layer_name}")  # Debugging print statement
    input_list = [i.detach().cpu().numpy().tolist() for i in input]
    output_list = output.detach().cpu().numpy().tolist()
    
    if layer_name not in model_dct:
        model_dct[layer_name] = []
    model_dct[layer_name].append({
        'input': input_list,
        'output': output_list
    })


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class TabularDataset(Dataset):
    def __init__(self, count_matrix, label):
        self.data = count_matrix
        self.features = self.data.values
        self.target = label.values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.target[idx], dtype=torch.float32)
        return features, target

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    predicted_list = []
    probability_list = []
    labels_list = []
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = 0
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            probability = torch.sigmoid(outputs.data)
            predicted = torch.round(torch.sigmoid(outputs.data))
            loss += criterion(outputs, labels)
            predicted_list.extend(predicted)
            labels_list.extend(labels)
            probability_list.extend(probability)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy, loss, predicted_list, labels_list, probability_list

def save_model(model_nn, model_path, model_state_dict_path):
    model_nn.eval()
    torch.save(model_nn, model_path)
    torch.save(model_nn.state_dict(), model_state_dict_path)

def train_model(train_dataloader, val_dataloader, test_dataloader, test_cell_id, layers_node, masking, output_layer, model_save_dir, date_string, learning_rate=0.001, num_epochs=50, weight_decay=0):
    model_nn = CustomNetwork(layers_node, output_layer, masking)
    optimizer = torch.optim.AdamW(model_nn.parameters(), lr=learning_rate, weight_decay=weight_decay)  
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    patience = 20
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    early_stop = False
    csv_file_path = f'{model_save_dir}{date_string}/training_log_{output_layer}.csv'

    try:
        os.makedirs(f'{model_save_dir}{date_string}')
    except:
        print(('...'))

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_accuracy', 'Validation_Loss', 'Val_accuracy'])

    for epoch in tqdm(range(num_epochs)):
        if early_stop:
            print("Early stopping")
            break
        
        total_loss = 0
        for batch_features, batch_targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model_nn(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
        
        train_accuracy, train_loss, _, _, _ = evaluate(model_nn, train_dataloader)
        val_accuracy, val_loss, _, _, _ = evaluate(model_nn, val_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Train_accuracy: {train_accuracy}, Val Loss: {val_loss.item():.4f}, Val_accuracy: {val_accuracy}')
        
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, loss.item(), train_accuracy, val_loss.item(), val_accuracy])
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            model_path = f'{model_save_dir}{date_string}/best_model_{output_layer}.pth'
            model_state_dict_path = f'{model_save_dir}{date_string}/best_model_{output_layer}_state_dict.pth'
            save_model(model_nn, model_path, model_state_dict_path)
            best_model_nn = copy.deepcopy(model_nn)
            print('Model saved.')
        else:
            epochs_no_improve += 1

    train_accuracy, train_loss, _, _, _ = evaluate(best_model_nn, train_dataloader)
    val_accuracy, val_loss, _, _, _ = evaluate(best_model_nn, val_dataloader)
    test_accuracy, test_loss, predicted_list_test, labels_list_test, test_probability_list = evaluate(best_model_nn, test_dataloader)
    print('Test Accuracy', test_accuracy)

    labels_list_test = [m.item() for m in labels_list_test]
    predicted_list_test = [m.item() for m in predicted_list_test]
    test_probability_list = [m.item() for m in test_probability_list]

    test_df = pd.DataFrame({'cell_id': test_cell_id, 'true_y': labels_list_test, 'pred_y': predicted_list_test, 'probability': test_probability_list})
    csv_file_path = f'{model_save_dir}{date_string}/test_log_{output_layer}.csv'
    test_df.to_csv(csv_file_path)
    return best_model_nn  # Return only the best model

def train_fc_model(train_dataloader, val_dataloader, test_dataloader, test_cell_id, layers_node, masking, output_layer, model_save_dir, date_string, learning_rate=0.001, num_epochs=50, weight_decay=0):
    model_nn = CustomfcNetwork(layers_node, output_layer, masking)
    optimizer = torch.optim.AdamW(model_nn.parameters(), lr=learning_rate, weight_decay=weight_decay)  
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    patience = 20
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    early_stop = False
    csv_file_path = f'{model_save_dir}{date_string}/fc_training_log_{output_layer}.csv'

    try:
        os.makedirs(f'{model_save_dir}{date_string}')
    except:
        print(('...'))

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_accuracy', 'Validation_Loss', 'Val_accuracy'])

    for epoch in tqdm(range(num_epochs)):
        if early_stop:
            print("Early stopping")
            break
        
        total_loss = 0
        for batch_features, batch_targets in train_dataloader:
            outputs = model_nn(batch_features)
            loss = criterion(outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_accuracy, train_loss, _, _, _ = evaluate(model_nn, train_dataloader)
        val_accuracy, val_loss, _, _, _ = evaluate(model_nn, val_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Train_accuracy: {train_accuracy}, Val Loss: {val_loss.item():.4f}, Val_accuracy: {val_accuracy}')
        
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, loss.item(), train_accuracy, val_loss.item(), val_accuracy])
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            model_path = f'{model_save_dir}{date_string}/fc_best_model_{output_layer}.pth'
            model_state_dict_path = f'{model_save_dir}{date_string}/fc_best_model_{output_layer}_state_dict.pth'
            save_model(model_nn, model_path, model_state_dict_path)
            best_model_nn = copy.deepcopy(model_nn)
            print('Model saved.')
        else:
            epochs_no_improve += 1

    train_accuracy, train_loss, _, _, _ = evaluate(best_model_nn, train_dataloader)
    val_accuracy, val_loss, _, _, _ = evaluate(best_model_nn, val_dataloader)
    test_accuracy, test_loss, predicted_list_test, labels_list_test, test_probability_list = evaluate(best_model_nn, test_dataloader)
    print('Test Accuracy', test_accuracy)

    labels_list_test = [m.item() for m in labels_list_test]
    predicted_list_test = [m.item() for m in predicted_list_test]
    test_probability_list = [m.item() for m in test_probability_list]

    test_df = pd.DataFrame({'cell_id': test_cell_id, 'true_y': labels_list_test, 'pred_y': predicted_list_test, 'probability': test_probability_list})
    csv_file_path = f'{model_save_dir}{date_string}/fc_test_log_{output_layer}.csv'
    test_df.to_csv(csv_file_path)
    return best_model_nn  # Return only the best model

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def calculate_and_plot_shap_values(model, dataloader, feature_names, cell_type_name, save_dir):
    """Calculate SHAP values using GradientExplainer and plot both summary and bar plots."""
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
    shap.summary_plot(shap_values, data, feature_names=feature_names, show=False, color=summary_color)
    plt.title(f"SHAP Summary Plot for {cell_type_name}")
    # Save the summary plot with the cell type name
    summary_plot_path = os.path.join(save_dir, f'{cell_type_name}_shap_summary_plot.png')
    plt.savefig(summary_plot_path)
    plt.show()  # Display the plot

    # Plot the SHAP bar plot (global feature importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, data, feature_names=feature_names, plot_type="bar", color=bar_color)
    plt.title(f"SHAP Feature Importance for {cell_type_name}")
    # Save the bar plot with the cell type name
    bar_plot_path = os.path.join(save_dir, f'{cell_type_name}_shap_feature_importance_plot.png')
    plt.savefig(bar_plot_path)
    plt.show()  # Display the plot

def main(config_file):
    config = load_config(config_file)
    
    train = pd.read_csv(config['dataset']['train'], index_col=0)
    test = pd.read_csv(config['dataset']['test'], index_col=0)
    val = pd.read_csv(config['dataset']['val'], index_col=0)
    y_train = pd.read_csv(config['dataset']['y_train'])
    y_test = pd.read_csv(config['dataset']['y_test'])
    y_val = pd.read_csv(config['dataset']['y_val'])
  
    r_data_tmp = train.T
    q_data_tmp = test.T
    v_data_tmp = val.T
    r_label_tmp = y_train

    print('Getting Marker Genes.......')
    train_x, test_x, val_x, train_y = get_expression(r_data_tmp, q_data_tmp, v_data_tmp, r_label_tmp,
                                                     thrh=config['gene_expression']['highly_expressed_threshold'],
                                                     thrl=config['gene_expression']['lowly_expressed_threshold'],
                                                     normalization=config['gene_expression']['normalization'],
                                                     marker=config['gene_expression']['marker'])
    
    print('Getting Pathway Genes.........')
    pathway_genes = get_gene_pathways(config['pathways_network']['ensemble_pathway_relation'], species=config['pathways_network']['species'])

    print('Getting Masking.........')
    masking, masking_df, layers_node, train_x, test_x, val_x = get_masking(config['pathways_network']['pathway_names'],
                                                                           pathway_genes,
                                                                           config['pathways_network']['pathway_relation'],
                                                                           train_x,
                                                                           test_x,
                                                                           val_x,
                                                                           train_y,
                                                                           config['pathways_network']['datatype'],
                                                                           config['pathways_network']['species'],
                                                                           config['pathways_network']['n_hidden_layer'])

    test_cell_id = list(test_x.T.index) 
    try:
        masking = list(masking.values())
        layers_node = list(layers_node.values())
    except:
        print('already_done')

    train_dataset = TabularDataset(train_x.T, train_y)
    val_dataset = TabularDataset(val_x.T, y_val)
    test_dataset = TabularDataset(test_x.T, y_test)  
    
    dataloader_params = {
        'batch_size': config['train']['batch_size'],
        'shuffle': False
    }

    train_dataloader = DataLoader(train_dataset, **dataloader_params)
    test_dataloader = DataLoader(test_dataset, **dataloader_params)
    val_dataloader = DataLoader(val_dataset, **dataloader_params)

    model_dict_sparse = dict()
    model_dict_fc = dict()
    activation_output = {}
    now = datetime.now()
    date_string = now.strftime("%Y_%m_%d_%H_%M_%S")

    try:
        os.makedirs(f"{config['model_output']['model_save_dir']}{date_string}")
    except:
        print(('...'))

    print('Training.........')
    for output_layer in range(2, len(masking) + 2):
        if config['gene_expression']['print_information']:
            print("Current sub-neural network has " + str(output_layer - 1) + " hidden layers.")
        model_dict_sparse[output_layer] = train_model(train_dataloader,
                                                      val_dataloader, test_dataloader, test_cell_id,
                                                      layers_node,
                                                      masking,
                                                      output_layer,
                                                      model_save_dir=config['model_output']['model_save_dir'], date_string=date_string,
                                                      learning_rate=config['train']['learning_rate'], num_epochs=config['train']['epochs'], weight_decay=config['train']['weight_decay']
                                                      )  

    print('Training fully connected layers:')
    for output_layer in range(2, len(masking) + 2):
        if config['gene_expression']['print_information']:
            print("Current sub-neural network has " + str(output_layer - 1) + " hidden layers.")
        model_dict_fc[output_layer] = train_fc_model(train_dataloader,
                                                     val_dataloader, test_dataloader, test_cell_id,
                                                     layers_node,
                                                     masking,
                                                     output_layer,
                                                     model_save_dir=config['model_output']['model_save_dir'], date_string=date_string,
                                                     learning_rate=config['train']['learning_rate'], num_epochs=config['train']['epochs'], weight_decay=config['train']['weight_decay']
                                                     )  
        
    new_parameter = {'date_string': date_string}
    config.update(new_parameter)
    save_path = f"{config['model_output']['model_save_dir']}{date_string}/config.yml"
    with open(save_path, 'w') as file:
        yaml.dump(config, file)

    for i in range(len(masking_df)):
        masking_df[i].to_csv(f"{config['model_output']['model_save_dir']}{date_string}/masking_df_{i}.csv")
    
    dataloader_params = {
        'batch_size': 1,
        'shuffle': False
    }

    train_dataloader = DataLoader(train_dataset, **dataloader_params)
    test_dataloader = DataLoader(test_dataset, **dataloader_params)
    val_dataloader = DataLoader(val_dataset, **dataloader_params)
    
    from functools import partial

    for j, model in model_dict_sparse.items():
        for name, layer in enumerate(model.children()):
            layer_name = 'fc' + str(name + 1)
                # Use partial to bind the layer_name to the hook function
            layer.register_forward_hook(partial(hook_fn, layer_name=layer_name))

        accuracy, loss, _, _, _ = evaluate(model, test_dataloader)
        print(f"Test Accuracy for model {j}: {accuracy}")   
        accuracy, loss, _, _, _ = evaluate(model, train_dataloader)
        print(f"Train Accuracy for model {j}: {accuracy}")   
        accuracy, loss, _, _, _ = evaluate(model, val_dataloader)
        print(f"Validation Accuracy for model {j}: {accuracy}") 

        with open(f'{config["model_output"]["model_save_dir"]}{date_string}/model_activations_train_test.csv', 'w', newline='') as csvfile:
            fieldnames = ['layer', 'input', 'output']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for layer_name, activations in model_dct.items():
                for activation in activations:
                    writer.writerow({
                        'layer': layer_name,
                        'input': activation['input'],
                        'output': activation['output']
                    })

        # After evaluation, calculate SHAP values
        feature_names = list(train_x.index)  # Assuming train_x is the dataframe with gene names as index
        cell_type_name = config['dataset']['train'].split('/')[-2]  # Extract cell type name from the directory structure
        calculate_and_plot_shap_values(model, test_dataloader, feature_names, cell_type_name, config['model_output']['model_save_dir'] + date_string)
        
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training for all cell types')
    parser.add_argument('--configs', type=str, nargs='+', help='List of configuration files', required=True)
    args = parser.parse_args()
    
    for config_file in args.configs:
        print(f"Starting training with {config_file}")
        main(config_file)
        print(f"Finished training with {config_file}\n")
