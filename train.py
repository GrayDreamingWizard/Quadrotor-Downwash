#!/usr/bin/env python3
#training of NN for dw prediction

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ml_models import DownwashPredictionNetwork, DownwashDataset
from ml_models import SimpleDownwashNetwork, DeepDownwashNetwork

def train_downwash_model(data_path='data/downwash_data.pkl', 
                         model_path='models/downwash_model.pth',
                         model_type='standard',
                         batch_size=64, 
                         epochs=100, 
                         lr=0.001):
    """
    training the nn to predict dw forces
    args: paths for data/model, batch_sz, epochs and learngin rate
    return: tuple of (model, scaler_X, scaler_Y)
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print("Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    targets = data['targets']
    
    print(f"Data loaded: {len(features)} samples")
    print(f"Feature dimensions: {features.shape}, Target dimensions: {targets.shape}")

    print("Normalizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    features_scaled = scaler_X.fit_transform(features)
    targets_scaled = scaler_y.fit_transform(targets)
    
    #splitting data into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, targets_scaled, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    train_dataset = DownwashDataset(X_train, y_train)
    test_dataset = DownwashDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    #model type initialization
    if model_type == 'simple':
        model = SimpleDownwashNetwork(input_dim=features.shape[1])
        print("Using simple network architecture")
    elif model_type == 'deep':
        model = DeepDownwashNetwork(input_dim=features.shape[1])
        print("Using deep network architecture")
    else:  
        model = DownwashPredictionNetwork(input_dim=features.shape[1])
        print("Using standard network architecture")
    
    #loss func and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
    
    train_losses = []
    test_losses = []
    
    print("Starting training...")
    #training loop and training phase
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
        
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        #eval phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        #learning rate update
        scheduler.step(test_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    print("Training complete")
    
    #final model performance eval
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        
        for inputs, targets in test_loader:
            outputs = model(inputs)
            
            #storing predictions and targets
            all_preds.append(outputs.numpy())
            all_targets.append(targets.numpy())
    
    #combining batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    #original scale using inv transform
    preds_orig = scaler_y.inverse_transform(all_preds)
    targets_orig = scaler_y.inverse_transform(all_targets)
    
    # Calculating metrics, saving models and scalers 
    mse = np.mean((preds_orig - targets_orig)**2, axis=0)
    mae = np.mean(np.abs(preds_orig - targets_orig), axis=0)
    
    print("\nFinal Model Evaluation:")
    print(f"MSE by component: X: {mse[0]:.6f}, Y: {mse[1]:.6f}, Z: {mse[2]:.6f}")
    print(f"MAE by component: X: {mae[0]:.6f}, Y: {mae[1]:.6f}, Z: {mae[2]:.6f}")
    
    print(f"Saving model to {model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'metrics': {
            'mse': mse,
            'mae': mae
        }
    }, model_path)
    
    # plotting training curves 
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Downwash Model Training Progress')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_curve.png')
    
    plt.figure(figsize=(15, 5))
    
    plot_indices = np.random.choice(len(preds_orig), min(1000, len(preds_orig)), replace=False)
    
    for i, component in enumerate(['X', 'Y', 'Z']):
        plt.subplot(1, 3, i+1)
        plt.scatter(targets_orig[plot_indices, i], preds_orig[plot_indices, i], alpha=0.5, s=10)
        min_val = min(targets_orig[:, i].min(), preds_orig[:, i].min())
        max_val = max(targets_orig[:, i].max(), preds_orig[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel(f'True Force {component} (N)')
        plt.ylabel(f'Predicted Force {component} (N)')
        plt.title(f'Force {component} Predictions')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/prediction_vs_target.png')
    
    print("Saved plots to plots/ directory")
    
    return model, scaler_X, scaler_y

def load_downwash_model(model_path='models/downwash_model.pth'):
    """
    loading trained dw model 
    args: model path
    return: tuple (model, scaler_X, scaler_y)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run training first.")
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path)
    
    model = DownwashPredictionNetwork(input_dim=12)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  
    
    return model, checkpoint['scaler_X'], checkpoint['scaler_y']

def train_model_variations():
    """
    if chosen ablation, train different models 
    """
    print("Training model variations for ablation studies...")
    
    #training diff model archs 
    for model_type in ['simple', 'standard', 'deep']:
        print(f"\n=== Training {model_type} model ===")
        train_downwash_model(
            model_path=f'models/downwash_model_{model_type}.pth',
            model_type=model_type,
            epochs=50  #few epochs for demo
        )
    
    # comparison to training results 
    model_results = {}
    for model_type in ['simple', 'standard', 'deep']:
        checkpoint = torch.load(f'models/downwash_model_{model_type}.pth')
        model_results[model_type] = {
            'train_losses': checkpoint['train_losses'],
            'test_losses': checkpoint['test_losses'],
            'metrics': checkpoint['metrics']
        }
    plt.figure(figsize=(12, 8))
    
    #Training loss comparison 
    plt.subplot(2, 1, 1)
    for model_type in model_results:
        plt.plot(model_results[model_type]['train_losses'], label=f'{model_type} - Train')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    #test loss comparison
    plt.subplot(2, 1, 2)
    for model_type in model_results:
        plt.plot(model_results[model_type]['test_losses'], label=f'{model_type} - Test')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    
    #printing final metrics 
    print("\n=== Model Comparison ===")
    for model_type in model_results:
        mse = model_results[model_type]['metrics']['mse']
        mae = model_results[model_type]['metrics']['mae']
        print(f"\n{model_type.title()} Model:")
        print(f"MSE (Z-component): {mse[2]:.6f}")
        print(f"MAE (Z-component): {mae[2]:.6f}")
    
    print("\nComparison plot saved to plots/model_comparison.png")

if __name__ == "__main__":
    train_downwash_model()