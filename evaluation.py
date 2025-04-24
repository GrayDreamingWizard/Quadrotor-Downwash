#!/usr/bin/env python3
"""
evaluation and comparison of physics-based
and ML-based downwash models.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from physics_sim import Drone, PhysicsDownwashModel, pid_controller
from ml_models import MLDownwashModel
from train import load_downwash_model

def run_comparative_simulation(model_type="physics", strength=0.3, offset=0.0, simulation_time=5.0):
    """
    run sim using either physics based or ML model
    input: model type, strenght dw, offset dw, sim time
    return: dict with sim results
    """
    print(f"running {model_type}-based simulation (strength={strength}, offset={offset})...")
    
    #sim params
    dt = 0.01  #dt: time step
    steps = int(simulation_time / dt)
    times = np.linspace(0, simulation_time, steps)
    
    #note: fixed upper drone at specific offset
    upper_pos = np.array([offset, 0, 1.0])
    upper_vel = np.zeros(3)
    
    #lower drone initialization
    lower_drone_with_comp = Drone(init_pos=np.array([0, 0, 0.0]))
    lower_drone_without_comp = Drone(init_pos=np.array([0, 0, 0.0]))
    
    # baseline quad, no dw
    lower_drone_baseline = Drone(init_pos=np.array([0, 0, 0.0]))
    
    #dw model based on the selected type
    params = {'strength': strength, 'width': 0.5, 'lateral_coeff': 0.05}
    
    if model_type == "physics":
        downwash_model = PhysicsDownwashModel(strength=strength)
    elif model_type == "ml":
        if not os.path.exists('models/downwash_model.pth'):
            raise FileNotFoundError("ML model not found. Run training first.")
        
        downwash_model = MLDownwashModel()
    else:
        raise ValueError("model_type must be 'physics' or 'ml'")
    
    #stack results
    lower_comp_positions = np.zeros((steps, 3))
    lower_comp_velocities = np.zeros((steps, 3))
    lower_no_comp_positions = np.zeros((steps, 3))
    lower_no_comp_velocities = np.zeros((steps, 3))
    lower_baseline_positions = np.zeros((steps, 3))
    downwash_forces = np.zeros((steps, 3))
    comp_thrusts = np.zeros(steps)
    no_comp_thrusts = np.zeros(steps)
    
    #saveing lower drone positions
    lower_comp_positions[0] = lower_drone_with_comp.pos
    lower_no_comp_positions[0] = lower_drone_without_comp.pos
    lower_baseline_positions[0] = lower_drone_baseline.pos
    
    #fixing lower drone target at 0
    lower_target = np.array([0, 0, 0.0])
    
    #PID controller state reset
    pid_controller.integral_error = np.zeros(3)
    
    #sim begin
    for i in range(1, steps):
        #dw force calculation
        if model_type == "physics":
            downwash_force = downwash_model.calculate_force(
                upper_pos, upper_vel, lower_drone_with_comp.pos, lower_drone_with_comp.vel
            )
        else: #use ml model if not physics based
            downwash_force = downwash_model.calculate_force(
                upper_pos, upper_vel, lower_drone_with_comp.pos, lower_drone_with_comp.vel, params
            )
        
        downwash_forces[i] = downwash_force
        
        #full compensation thrust ccalculation
        comp_thrust = -downwash_force[2] * lower_drone_with_comp.mass * 10.0
        comp_thrusts[i] = -downwash_force[2] * lower_drone_with_comp.mass 
        
        #lower drone cntrl with and without compensation
        lower_comp_base_thrust = pid_controller(
            lower_target, lower_drone_with_comp.pos, lower_drone_with_comp.vel, dt
        )
        lower_comp_thrust = lower_comp_base_thrust + comp_thrust
        comp_thrusts[i] = comp_thrust
        
        lower_no_comp_thrust = pid_controller(
            lower_target, lower_drone_without_comp.pos, lower_drone_without_comp.vel, dt
        )
        no_comp_thrusts[i] = lower_no_comp_thrust
        
        #baseline drone case with no downwash 
        lower_baseline_thrust = pid_controller(
            lower_target, lower_drone_baseline.pos, lower_drone_baseline.vel, dt
        )
        
        #state updates below
        # lower drone state update w/ compensation
        lower_comp_positions[i], lower_comp_velocities[i] = lower_drone_with_comp.update(
            lower_comp_thrust, downwash_force, dt
        )
        
        # lower drone state update w/o compensation
        lower_no_comp_positions[i], lower_no_comp_velocities[i] = lower_drone_without_comp.update(
            lower_no_comp_thrust, downwash_force, dt
        )
        
        #Update baseline drone (no downwash)
        lower_drone_baseline.update(lower_baseline_thrust, np.zeros(3), dt)
        lower_baseline_positions[i] = lower_drone_baseline.pos
        
        if i % (steps // 5) == 0:
            print(f"Time: {times[i]:.1f}s - Lower with comp: {lower_comp_positions[i, 2]:.2f}m, " +
                  f"Lower without comp: {lower_no_comp_positions[i, 2]:.2f}m")
            print(f"Downwash Z force: {downwash_force[2]:.4f}N, Compensation: {comp_thrust:.4f}N")
    
    #RMSE to target position calculation
    baseline_rmse = np.sqrt(np.mean((lower_baseline_positions[:, 2] - lower_target[2])**2))
    lower_comp_rmse = np.sqrt(np.mean((lower_comp_positions[:, 2] - lower_target[2])**2))
    lower_no_comp_rmse = np.sqrt(np.mean((lower_no_comp_positions[:, 2] - lower_target[2])**2))
    
    # improvement percentage calculation
    downwash_effect = (lower_no_comp_rmse - baseline_rmse) / baseline_rmse * 100
    compensation_improvement = (lower_no_comp_rmse - lower_comp_rmse) / lower_no_comp_rmse * 100
    
    print("Simulation complete!")
    print(f"Baseline RMSE (no downwash): {baseline_rmse:.4f}m")
    print(f"RMSE with comp: {lower_comp_rmse:.4f}m, RMSE without comp: {lower_no_comp_rmse:.4f}m")
    print(f"Downwash effect: +{downwash_effect:.1f}% error")
    print(f"Compensation improvement: {compensation_improvement:.1f}%")
    
    return {
        'times': times,
        'upper_pos': upper_pos,
        'lower_comp_positions': lower_comp_positions,
        'lower_no_comp_positions': lower_no_comp_positions,
        'lower_baseline_positions': lower_baseline_positions,
        'downwash_forces': downwash_forces,
        'comp_thrusts': comp_thrusts,
        'metrics': {
            'baseline_rmse': baseline_rmse,
            'lower_comp_rmse': lower_comp_rmse,
            'lower_no_comp_rmse': lower_no_comp_rmse,
            'downwash_effect': downwash_effect,
            'compensation_improvement': compensation_improvement
        },
        'params': {
            'strength': strength,
            'offset': offset,
            'model_type': model_type
        }
    }

def plot_results(result):
    """
    Create plots for simulation results
    
    Args:
        result: Dictionary with simulation results
    """
    os.makedirs('plots', exist_ok=True)
    
    strength = result['params']['strength']
    offset = result['params']['offset']
    model_type = result['params']['model_type']
    
    plt.figure(figsize=(12, 8))
    
    #z position plot
    plt.subplot(2, 1, 1)
    plt.plot(result['times'], result['lower_baseline_positions'][:, 2], 'b-', 
             label='Baseline (No Downwash)')
    plt.plot(result['times'], result['lower_comp_positions'][:, 2], 'g-', 
             label='With Compensation')
    plt.plot(result['times'], result['lower_no_comp_positions'][:, 2], 'r-', 
             label='Without Compensation')
    plt.axhline(y=0.0, color='k', linestyle=':', alpha=0.5, label='Target Height')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.title(f'{model_type.title()}-Based Downwash Simulation (Strength={strength}, Offset={offset}m)')
    plt.legend()
    plt.grid(True)
    
    #force and thrust plots
    plt.subplot(2, 1, 2)
    plt.plot(result['times'], -result['downwash_forces'][:, 2], 'r-', label='downwash Force')
    plt.plot(result['times'], result['comp_thrusts'], 'g-', label='compensation Thrust')
    plt.xlabel('Time (s)')
    plt.ylabel('Force/Thrust (N)')
    plt.title('Downwash Forces and Compensation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_type}_simulation.png')
    
    print(f"Saved plot to plots/{model_type}_simulation.png")
    
    #bar chart for metric comparisons
    plt.figure(figsize=(10, 6))
    
    metrics = [
        result['metrics']['baseline_rmse'],
        result['metrics']['lower_no_comp_rmse'],
        result['metrics']['lower_comp_rmse']
    ]
    
    labels = [
        'baseline\n(no downwash)', 
        'with downwash\nno compensation', 
        'with downwash\nand compensation'
    ]
    
    plt.bar(labels, metrics, color=['blue', 'red', 'green'])
    plt.ylabel('RMSE (m)')
    plt.title(f'{model_type.title()} Model Performance Metrics')
    
    # Add improvement percentages as text
    for i, m in enumerate(metrics):
        plt.text(i, m + 0.002, f"{m:.4f}m", ha='center')
    
    # Add a text box with the improvement percentage
    downwash_effect = f"Downwash Effect: +{result['metrics']['downwash_effect']:.1f}% error"
    compensation_effect = f"Compensation Improvement: {result['metrics']['compensation_improvement']:.1f}%"
    
    plt.annotate(downwash_effect + '\n' + compensation_effect,
                xy=(0.5, 0.9),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_type}_metrics.png')

def ablation_study():
    """
    compariison of physics based and model based across a range of params
    return: dict w/ all sim results
    """
    print("running ablation study...")
    
    #param ranges and results stack
    strengths = [0.1, 0.2, 0.3, 0.4]
    offsets = [0.0, 0.2, 0.4, 0.6]
    model_types = ["physics", "ml"]
    
    results = {
        "physics": [],
        "ml": []
    }
    
    for model_type in model_types:
        for strength in strengths:
            for offset in offsets:
                print(f"\nTesting {model_type} model with strength={strength}, offset={offset}m...")
                result = run_comparative_simulation(model_type, strength, offset)
                results[model_type].append(result)
    
    physics_downwash = np.zeros((len(strengths), len(offsets)))
    physics_improvement = np.zeros((len(strengths), len(offsets)))
    ml_downwash = np.zeros((len(strengths), len(offsets)))
    ml_improvement = np.zeros((len(strengths), len(offsets)))
    
    for i, strength in enumerate(strengths):
        for j, offset in enumerate(offsets):
            for result in results["physics"]:
                if (result['params']['strength'] == strength and 
                    result['params']['offset'] == offset):
                    physics_downwash[i, j] = result['metrics']['downwash_effect']
                    physics_improvement[i, j] = result['metrics']['compensation_improvement']
            
            for result in results["ml"]:
                if (result['params']['strength'] == strength and 
                    result['params']['offset'] == offset):
                    ml_downwash[i, j] = result['metrics']['downwash_effect']
                    ml_improvement[i, j] = result['metrics']['compensation_improvement']
    
    plt.figure(figsize=(15, 10))
    
    #physics-based model dw heat map
    plt.subplot(2, 2, 1)
    plt.imshow(physics_downwash, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Downwash Effect (%)')
    plt.xticks(np.arange(len(offsets)), [f"{o}m" for o in offsets])
    plt.yticks(np.arange(len(strengths)), [f"{s}" for s in strengths])
    plt.xlabel('Horizontal Offset')
    plt.ylabel('Downwash Strength')
    plt.title('Physics-Based Model Downwash Effect')
    
    for i in range(len(strengths)):
        for j in range(len(offsets)):
            plt.text(j, i, f"{physics_downwash[i, j]:.1f}%", 
                     ha="center", va="center", color="w")
    
    #physics based improvement map
    plt.subplot(2, 2, 2)
    plt.imshow(physics_improvement, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Improvement (%)')
    plt.xticks(np.arange(len(offsets)), [f"{o}m" for o in offsets])
    plt.yticks(np.arange(len(strengths)), [f"{s}" for s in strengths])
    plt.xlabel('Horizontal Offset')
    plt.ylabel('Downwash Strength')
    plt.title('Physics-Based Model Improvement')
    
    for i in range(len(strengths)):
        for j in range(len(offsets)):
            plt.text(j, i, f"{physics_improvement[i, j]:.1f}%", 
                     ha="center", va="center", color="w")
    
    #ml-based model dw heat map
    plt.subplot(2, 2, 3)
    plt.imshow(ml_downwash, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Downwash Effect (%)')
    plt.xticks(np.arange(len(offsets)), [f"{o}m" for o in offsets])
    plt.yticks(np.arange(len(strengths)), [f"{s}" for s in strengths])
    plt.xlabel('Horizontal Offset')
    plt.ylabel('Downwash Strength')
    plt.title('ML-Based Model Downwash Effect')
    
    for i in range(len(strengths)):
        for j in range(len(offsets)):
            plt.text(j, i, f"{ml_downwash[i, j]:.1f}%", 
                     ha="center", va="center", color="w")
    
     # ml based improvement map
    plt.subplot(2, 2, 4)
    plt.imshow(ml_improvement, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Improvement (%)')
    plt.xticks(np.arange(len(offsets)), [f"{o}m" for o in offsets])
    plt.yticks(np.arange(len(strengths)), [f"{s}" for s in strengths])
    plt.xlabel('Horizontal Offset')
    plt.ylabel('Downwash Strength')
    plt.title('ML-Based Model Improvement')
    
    for i in range(len(strengths)):
        for j in range(len(offsets)):
            plt.text(j, i, f"{ml_improvement[i, j]:.1f}%", 
                     ha="center", va="center", color="w")
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison_heatmap.png')
    
    print("saved comparison heatmap to plots/model_comparison_heatmap.png")
    
    #direct comparison
    strength = 0.3
    offset = 0.0
    
    physics_result = None
    ml_result = None
    
    for result in results["physics"]:
        if (result['params']['strength'] == strength and 
            result['params']['offset'] == offset):
            physics_result = result
    
    for result in results["ml"]:
        if (result['params']['strength'] == strength and 
            result['params']['offset'] == offset):
            ml_result = result
    
    if physics_result and ml_result:
        plt.figure(figsize=(15, 10))
        
        #dw force comparison
        plt.subplot(2, 1, 1)
        plt.plot(physics_result['times'], -physics_result['downwash_forces'][:, 2], 'r-', 
                 label='Physics-Based Downwash Force')
        plt.plot(ml_result['times'], -ml_result['downwash_forces'][:, 2], 'b-', 
                 label='ML-Based Downwash Force')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.title(f'Downwash Force Comparison (Strength={strength}, Offset={offset}m)')
        plt.legend()
        plt.grid(True)
        
        # drone z pos comparison
        plt.subplot(2, 1, 2)
        plt.plot(physics_result['times'], physics_result['lower_baseline_positions'][:, 2], 'k-', 
                 label='Baseline')
        plt.plot(physics_result['times'], physics_result['lower_comp_positions'][:, 2], 'g-', 
                 label='Physics-Based with Comp')
        plt.plot(physics_result['times'], physics_result['lower_no_comp_positions'][:, 2], 'r-', 
                 label='Physics-Based without Comp')
        plt.plot(ml_result['times'], ml_result['lower_comp_positions'][:, 2], 'g--', 
                 label='ML-Based with Comp')
        plt.plot(ml_result['times'], ml_result['lower_no_comp_positions'][:, 2], 'r--', 
                 label='ML-Based without Comp')
        plt.axhline(y=0.0, color='k', linestyle=':', alpha=0.5, label='Target Height')
        plt.xlabel('Time (s)')
        plt.ylabel('Z Position (m)')
        plt.title(f'Drone Height Comparison (Strength={strength}, Offset={offset}m)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/physics_vs_ml_comparison.png')
        
        print("Saved detailed comparison to plots/physics_vs_ml_comparison.png")
    
    #comparison tables 
    print("\n=== Performance Metrics Comparison ===")
    print("\nPhysics-Based Model (Downwash Effect / Compensation Improvement):")
    for i, strength in enumerate(strengths):
        print(f"  Strength {strength}: {', '.join([f'{offset}m: {physics_downwash[i, j]:.1f}% / {physics_improvement[i, j]:.1f}%' for j, offset in enumerate(offsets)])}")
    
    print("\nML-Based Model (Downwash Effect / Compensation Improvement):")
    for i, strength in enumerate(strengths):
        print(f"  Strength {strength}: {', '.join([f'{offset}m: {ml_downwash[i, j]:.1f}% / {ml_improvement[i, j]:.1f}%' for j, offset in enumerate(offsets)])}")
    
    return results

def model_evaluation():
    """
    Detailed evaluation of the machine learning model performance
    """
    print("Running detailed ML model evaluation...")
    
    #load trained model and scalers
    model, scaler_X, scaler_y = load_downwash_model()
    
    strengths = [0.1, 0.2, 0.3, 0.4]
    heights = [0.2, 0.4, 0.6, 0.8, 1.0]
    offsets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    physics_model = PhysicsDownwashModel()
    
    test_points = []
    physics_forces = []
    ml_forces = []
    
    for strength in strengths:
        physics_model.strength = strength
        
        for height in heights:
            for offset in offsets:
                upper_pos = np.array([offset, 0, height])
                upper_vel = np.zeros(3)
                lower_pos = np.array([0, 0, 0])
                lower_vel = np.zeros(3)
                
                physics_force = physics_model.calculate_force(
                    upper_pos, upper_vel, lower_pos, lower_vel
                )
                
                params = {'strength': strength, 'width': 0.5, 'lateral_coeff': 0.05}
                
                ml_model = MLDownwashModel()
                
                ml_force = ml_model.calculate_force(
                    upper_pos, upper_vel, lower_pos, lower_vel, params
                )
                
                test_points.append((strength, height, offset))
                physics_forces.append(physics_force)
                ml_forces.append(ml_force)
    
    test_points = np.array(test_points)
    physics_forces = np.array(physics_forces)
    ml_forces = np.array(ml_forces)
    #error calc w/ numerical stability
    errors = np.abs(ml_forces - physics_forces)
    relative_errors = errors / (np.abs(physics_forces) + 1e-6)  

    # stats calc
    mean_abs_error = np.mean(errors, axis=0)
    mean_rel_error = np.mean(relative_errors, axis=0)
    
    print("\n=== ML Model Prediction Error Analysis ===")
    print(f"Mean Absolute Error (X, Y, Z): {mean_abs_error}")
    print(f"Mean Relative Error (X, Y, Z): {mean_rel_error}")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(errors[:, 2], bins=30, alpha=0.7)
    plt.xlabel('Absolute Error (N)')
    plt.ylabel('Count')
    plt.title('Z-Force Prediction Error Distribution')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.scatter(test_points[:, 1], errors[:, 2], alpha=0.5, s=10)
    plt.xlabel('Vertical Distance (m)')
    plt.ylabel('Z-Force Error (N)')
    plt.title('Error vs Vertical Distance')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.scatter(test_points[:, 0], errors[:, 2], alpha=0.5, s=10)
    plt.xlabel('Downwash Strength')
    plt.ylabel('Z-Force Error (N)')
    plt.title('Error vs Downwash Strength')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.scatter(test_points[:, 2], errors[:, 2], alpha=0.5, s=10)
    plt.xlabel('Horizontal Offset (m)')
    plt.ylabel('Z-Force Error (N)')
    plt.title('Error vs Horizontal Offset')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/ml_model_error_analysis.png')
    
    print("Saved error analysis plots to plots/ml_model_error_analysis.png")
    
    #force field viz
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        #initializing params
        physics_model.strength = 0.3
        params = {'strength': 0.3, 'width': 0.5, 'lateral_coeff': 0.05}
        
        x = np.linspace(-0.5, 0.5, 10)
        z = np.linspace(0.2, 1.0, 10)
        X, Z = np.meshgrid(x, z)
        
        physics_forces = np.zeros((X.shape[0], X.shape[1], 3))
        ml_forces = np.zeros((X.shape[0], X.shape[1], 3))
        
        upper_vel = np.zeros(3)
        lower_vel = np.zeros(3)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                upper_pos = np.array([X[i, j], 0, Z[i, j] + 0.5])  #forcing upper drone to be 0.5 m z above
                lower_pos = np.array([X[i, j], 0, Z[i, j]])
                
                physics_forces[i, j] = physics_model.calculate_force(
                    upper_pos, upper_vel, lower_pos, lower_vel
                )
                
                ml_forces[i, j] = ml_model.calculate_force(
                    upper_pos, upper_vel, lower_pos, lower_vel, params
                )

        fig = plt.figure(figsize=(15, 7))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.quiver(X, np.zeros_like(X), Z, physics_forces[:, :, 0], np.zeros_like(X), physics_forces[:, :, 2],length=0.1, normalize=True, color='r')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('Z Position (m)')
        ax1.set_title('Physics-Based Downwash Force Field')
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.quiver(X, np.zeros_like(X), Z, ml_forces[:, :, 0], np.zeros_like(X), ml_forces[:, :, 2],length=0.1, normalize=True, color='b')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_zlabel('Z Position (m)')
        ax2.set_title('ML-Based Downwash Force Field')
        
        plt.tight_layout()
        plt.savefig('plots/force_field_visualization.png')
        
        print("Saved force field visualization to plots/force_field_visualization.png")
    
    except ImportError:
        print("3D visualization requires mpl_toolkits.mplot3d, skipping...")
    
    return {
        'test_points': test_points,
        'physics_forces': physics_forces,
        'ml_forces': ml_forces,
        'errors': errors,
        'mean_abs_error': mean_abs_error,
        'mean_rel_error': mean_rel_error
    }

if __name__ == "__main__":
    results = run_comparative_simulation(model_type="physics")
    plot_results(results)