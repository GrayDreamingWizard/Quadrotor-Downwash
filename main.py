#!/usr/bin/env python3
"""
main driver script to work b/w physics based and ml based
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from data_generator import generate_downwash_data
from ml_models import MLDownwashModel
from physics_sim import Drone, PhysicsDownwashModel, pid_controller
from train import train_downwash_model, load_downwash_model
from evaluation import run_comparative_simulation, plot_results, ablation_study
from visualization import create_animation

def main():
    """Main execution function for the ML-Enhanced Downwash Simulation"""
    for directory in ['data', 'models', 'plots']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    parser = argparse.ArgumentParser(description='ML-Enhanced Drone Downwash Simulation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'ablation', 'visualize'],help='Execution mode')
    parser.add_argument('--model', type=str, default='ml',choices=['physics', 'ml'],help='Model type to use')
    parser.add_argument('--strength', type=float, default=0.3,help='Downwash strength parameter')
    parser.add_argument('--offset', type=float, default=0.0,help='Horizontal offset between drones (m)')
    parser.add_argument('--num_samples', type=int, default=10000,help='Number of data samples to generate for training')
    parser.add_argument('--epochs', type=int, default=100,help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("=== Training ML Downwash Model ===")
        
        if not os.path.exists('data/downwash_data.pkl'):
            print("Generating training data...")
            generate_downwash_data(num_samples=args.num_samples)
        else:
            print("Using existing training data...")
        
        #train model 
        print("Training neural network model...")
        train_downwash_model(epochs=args.epochs)
        
        print("Training complete! Model saved to models/downwash_model.pth")
        
        #model eval
    elif args.mode == 'evaluate':
        print(f"=== Evaluating {args.model.upper()} Model ===")
        print(f"Parameters: strength={args.strength}, offset={args.offset}m")
        
        result = run_comparative_simulation(
            model_type=args.model,
            strength=args.strength,
            offset=args.offset
        )
        
        # plotting results and metrics
        plot_results(result)
        
        print("\n=== Performance Metrics ===")
        print(f"RMSE with compensation: {result['metrics']['lower_comp_rmse']:.4f}m")
        print(f"RMSE without compensation: {result['metrics']['lower_no_comp_rmse']:.4f}m")
        print(f"Baseline RMSE (no downwash): {result['metrics']['baseline_rmse']:.4f}m")
        print(f"Downwash effect: +{result['metrics']['downwash_effect']:.2f}% error")
        print(f"Compensation improvement: {result['metrics']['compensation_improvement']:.2f}%")
        create_animation(result)
        
    elif args.mode == 'ablation':
        print("=== Running Ablation Study ===")
        print("Testing multiple strengths and offsets with both physics and ML models")
        ablation_results = ablation_study()
        
    elif args.mode == 'visualize':
        print("=== Creating Visualizations ===")
        physics_result = run_comparative_simulation(
            model_type='physics',
            strength=args.strength,
            offset=args.offset
        )
        
        ml_result = run_comparative_simulation(
            model_type='ml',
            strength=args.strength,
            offset=args.offset
        )
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(physics_result['times'], -physics_result['downwash_forces'][:, 2], 'r-',label='Physics-Based Downwash Force')
        plt.plot(ml_result['times'], -ml_result['downwash_forces'][:, 2], 'b-',label='ML-Based Downwash Force')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.title(f'Downwash force comparison (Strength={args.strength}, Offset={args.offset}m)')
        plt.legend()
        plt.grid(True)
        
        #parse drone heights and compare
        plt.subplot(2, 1, 2)
        plt.plot(physics_result['times'], physics_result['lower_baseline_positions'][:, 2], 'k-',label='Baseline (No Downwash)')
        plt.plot(physics_result['times'], physics_result['lower_comp_positions'][:, 2], 'g-', label='Physics-Based with Comp')
        plt.plot(physics_result['times'], physics_result['lower_no_comp_positions'][:, 2], 'r-',label='Physics-Based without Comp')
        plt.plot(ml_result['times'], ml_result['lower_comp_positions'][:, 2], 'g--',label='ML-Based with Comp')
        plt.plot(ml_result['times'], ml_result['lower_no_comp_positions'][:, 2], 'r--',label='ML-Based without Comp')
        plt.axhline(y=0.0, color='k', linestyle=':', alpha=0.5, label='Target Height')
        plt.xlabel('Time (s)')
        plt.ylabel('Z Position (m)')
        plt.title(f'Drone height comparison (Strength={args.strength}, Offset={args.offset}m)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/physics_vs_ml_comparison.png')

if __name__ == "__main__":
    main()