#!/usr/bin/env python3
#nn arch for prediction of dw forces

import torch
import torch.nn as nn
import numpy as np
import os

class DownwashPredictionNetwork(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64):
        super(DownwashPredictionNetwork, self).__init__()
        
        #network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            # force vector (x,y,z)
            nn.Linear(hidden_dim//2, 3)  
        )
        
        #intiializing weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass;
        args, x: Tensor:
               - relative position (3)
               - upper drone velocity (3)
               - lower drone velocity (3)
               - downwash parameters (3): [strength, width, lateral_coeff]
        
        returns: Predicted force vector (3): [force_x, force_y, force_z]
        """
        return self.network(x)

class MLDownwashModel:
    #nn based dw model
    def __init__(self, model_path='models/downwash_model.pth'):
        #loading trained network by intializing ML dw model 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Run training first.")
        
        #load models, scalers and creates models
        checkpoint = torch.load(model_path)
        self.model = DownwashPredictionNetwork(input_dim=12)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval() 
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
    
    def calculate_force(self, upper_pos, upper_vel, lower_pos, lower_vel, params=None):
        #returns dw force in 3d: [fx, fy, fz], taking inputs upper & lower pos, upper & lower vel, and params
        if params is None:
            params = {'strength': 0.2, 'width': 0.5, 'lateral_coeff': 0.05}
        
        rel_pos = upper_pos - lower_pos
        
        #check to ensure lower below upper to apply dw
        if rel_pos[2] <= 0:
            return np.zeros(3)
        
        feature = np.concatenate([
            rel_pos,               
            upper_vel,              
            lower_vel,              
            [params['strength'], params['width'], params['lateral_coeff']]  
        ])
        
        feature_scaled = self.scaler_X.transform(feature.reshape(1, -1))
        
        with torch.no_grad():
            force_scaled = self.model(torch.tensor(feature_scaled, dtype=torch.float32)).numpy()
        
        force = self.scaler_y.inverse_transform(force_scaled)
        
        return force[0]

class DownwashDataset(torch.utils.data.Dataset):
    #intializing dataset for dw prediction: features and target args
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class SimpleDownwashNetwork(nn.Module):
    def __init__(self, input_dim=12):
        super(SimpleDownwashNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

class DeepDownwashNetwork(nn.Module):
    def __init__(self, input_dim=12):
        super(DeepDownwashNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    print("Testing DownwashPredictionNetwork...")
    
    model = DownwashPredictionNetwork(input_dim=12)
    
    x = torch.randn(10, 12)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("test successful!")