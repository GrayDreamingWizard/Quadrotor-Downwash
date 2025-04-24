#!/usr/bin/env python3
"""
Module to generate data for ML models by running physics-based sims with varying params
"""

import numpy as np
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

#import physics model
from physics_sim import PhysicsDownwashModel

def generate_downwash_data(num_samples=10000, save_path='data/downwash_data.pkl'):
    """
    generating training data for dw pred by sampling params, positions and calculating forces using physics based model
    num samples: num data points
    save path: generated data saved path

    retruns: features, targets tuple as numpy array
    """
    print(f"Generating {num_samples} downwash training samples...")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    features = []
    targets = []
    
    #setting param ranges
    strengths = np.random.uniform(0.05, 0.4, num_samples)
    widths = np.random.uniform(0.3, 0.8, num_samples)
    lateral_coeffs = np.random.uniform(0.01, 0.1, num_samples)
    
    progress_bar = tqdm(total=num_samples, desc="Generating data")
    
    #generating data, rndm rel positions and velocities
    for i in range(num_samples):
        rel_x = np.random.uniform(-1.0, 1.0)
        rel_y = np.random.uniform(-1.0, 1.0)
        rel_z = np.random.uniform(0.1, 1.5)  
        
        upper_vx = np.random.uniform(-0.5, 0.5)
        upper_vy = np.random.uniform(-0.5, 0.5)
        upper_vz = np.random.uniform(-0.5, 0.5)
        
        lower_vx = np.random.uniform(-0.5, 0.5)
        lower_vy = np.random.uniform(-0.5, 0.5)
        lower_vz = np.random.uniform(-0.5, 0.5)
        
        upper_pos = np.array([rel_x, rel_y, rel_z]) + np.array([0, 0, 0])
        upper_vel = np.array([upper_vx, upper_vy, upper_vz])
        lower_pos = np.array([0, 0, 0])
        lower_vel = np.array([lower_vx, lower_vy, lower_vz])
        
        model = PhysicsDownwashModel(
            strength=strengths[i],
            width=widths[i],
            lateral_coeff=lateral_coeffs[i]
        )
        
        # downwash force calc
        force = model.calculate_force(
            upper_pos, upper_vel, lower_pos, lower_vel
        )
        
        #feature vector
        feature = np.concatenate([
            upper_pos - lower_pos,  #relative position
            upper_vel,              #upper drone velocity
            lower_vel,              # Lower drone velocity
            [strengths[i], widths[i], lateral_coeffs[i]]  #parameters
        ])
        
        features.append(feature)
        targets.append(force)
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    #np array conversion and save data
    features = np.array(features)
    targets = np.array(targets)
    with open(save_path, 'wb') as f:
        pickle.dump({
            'features': features,
            'targets': targets,
            'feature_names': [
                'rel_x', 'rel_y', 'rel_z', 
                'upper_vx', 'upper_vy', 'upper_vz',
                'lower_vx', 'lower_vy', 'lower_vz',
                'strength', 'width', 'lateral_coeff'
            ],
            'target_names': ['force_x', 'force_y', 'force_z']
        }, f)
    
    print(f"Data generation complete! Saved {num_samples} samples to {save_path}")
    print(f"Feature shape: {features.shape}, Target shape: {targets.shape}")
    
    #visualization of data distribution
    visualize_data_distribution(features, targets, strengths, widths)
    
    return features, targets

def visualize_data_distribution(features, targets, strengths, widths):
    """Create visualizations of the generated data distribution"""
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    #plot on param distribution
    plt.subplot(2, 2, 1)
    plt.hist(strengths, bins=30, alpha=0.7)
    plt.xlabel('Downwash Strength')
    plt.ylabel('Count')
    plt.title('Distribution of Downwash Strengths')
    
    plt.subplot(2, 2, 2)
    plt.hist(widths, bins=30, alpha=0.7)
    plt.xlabel('Downwash Width')
    plt.ylabel('Count')
    plt.title('Distribution of Downwash Widths')
    
    #plot for relative pos
    plt.subplot(2, 2, 3)
    plt.scatter(features[:, 0], features[:, 1], alpha=0.1, s=1)
    plt.xlabel('Relative X Position (m)')
    plt.ylabel('Relative Y Position (m)')
    plt.title('Distribution of Horizontal Positions')
    plt.axis('equal')
    
    #plot for dw force, z component
    plt.subplot(2, 2, 4)
    plt.hist(targets[:, 2], bins=50, alpha=0.7) 
    plt.xlabel('Downwash Force Z (N)')
    plt.ylabel('Count')
    plt.title('Distribution of Downwash Force (Z)')
    
    plt.tight_layout()
    plt.savefig('plots/data_distribution.png')
    
    #scatter plot: vertical distance vs dw force
    plt.figure(figsize=(10, 6))
    plt.scatter(features[:, 2], -targets[:, 2], alpha=0.1, s=1)
    plt.xlabel('Vertical Distance (m)')
    plt.ylabel('Downwash Force Magnitude (N)')
    plt.title('Downwash Force vs. Vertical Distance')
    plt.grid(True)
    plt.savefig('plots/force_vs_distance.png')
    print("Created visualizations of data distribution in plots/ directory")

def load_downwash_data(data_path='data/downwash_data.pkl'):
    """
    load prev generated dw data
    data path: pth frm data file and returns dictionary (features, targets, metadata)
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded data from {data_path}")
    print(f"Feature shape: {data['features'].shape}, Target shape: {data['targets'].shape}")
    
    return data

if __name__ == "__main__":
    features, targets = generate_downwash_data(num_samples=5000)