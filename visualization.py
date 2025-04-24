#!/usr/bin/env python3
#animations and viz for drone dw sims

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

def create_animation(result, filename=None):
    """
    animation that takes in dict with sim resutls and returns confirmation
    """
    try:
        #extract data from resutls and add baseline positions 
        times = result['times']
        upper_pos = result['upper_pos']
        lower_comp_positions = result['lower_comp_positions']
        lower_no_comp_positions = result['lower_no_comp_positions']
        
        if 'lower_baseline_positions' in result:
            lower_baseline_positions = result['lower_baseline_positions']
            has_baseline = True
        else:
            has_baseline = False
            
        downwash_forces = result['downwash_forces']
        comp_thrusts = result['comp_thrusts']
        strength = result['params']['strength']
        offset = result['params']['offset']
        model_type = result['params']['model_type']
        
        if filename is None:
            filename = f'plots/downwash_animation_{model_type}_s{strength}_o{offset}.gif'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        print(f"Creating animation for {model_type} model (strength={strength}, offset={offset})...")
        
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
        
        ax1 = plt.subplot(gs[0, :2])
        ax1.set_xlim(-1.0, 1.0)
        ax1.set_ylim(-0.5, 1.5)
        
        #z position vs time
        ax2 = plt.subplot(gs[0, 2])
        ax2.set_xlim(0, times[-1])
        ax2.set_ylim(-0.5, 1.5)
        
        # force plot
        ax3 = plt.subplot(gs[1, :])
        ax3.set_xlim(0, times[-1])
        max_force = max(np.max(np.abs(downwash_forces[:, 2])), np.max(comp_thrusts)) * 1.2
        ax3.set_ylim(-max_force, max_force)
        ax1.axhline(y=0.0, color='k', linestyle=':', alpha=0.5)
        
        ax2.axhline(y=0.0, color='k', linestyle=':', alpha=0.5, label='Target')
        ax2.axhline(y=upper_pos[2], color='b', linestyle=':', alpha=0.5, label='Upper Drone')
        
        #drone figure as circles
        upper_drone = Circle((upper_pos[0], upper_pos[2]), 0.05, color='blue', label='Upper Drone')
        lower_drone_comp = Circle((0, lower_comp_positions[0, 2]), 0.05, color='green', label='Lower with Comp')
        lower_drone_no_comp = Circle((0.2, lower_no_comp_positions[0, 2]), 0.05, color='red', label='Lower without Comp')       
        ax1.add_patch(upper_drone)
        ax1.add_patch(lower_drone_comp)
        ax1.add_patch(lower_drone_no_comp)
        
        #add baseline value 
        if has_baseline:
            lower_drone_baseline = Circle((-0.2, lower_baseline_positions[0, 2]), 0.05, color='black', label='Baseline')
            ax1.add_patch(lower_drone_baseline)
        
        # arrow viz for forces 
        downwash_arrow = ax1.arrow(0, 0, 0, 0, head_width=0.03, head_length=0.05, fc='r', ec='r', alpha=0.7)
        comp_arrow = ax1.arrow(0, 0, 0, 0, head_width=0.03, head_length=0.05, fc='g', ec='g', alpha=0.7)
        
        if has_baseline:
            baseline_line, = ax2.plot([], [], 'k-', label='Baseline')
        comp_line, = ax2.plot([], [], 'g-', label='With Comp')
        no_comp_line, = ax2.plot([], [], 'r-', label='Without Comp')
        downwash_line, = ax3.plot([], [], 'r-', label='Downwash Force')
        comp_thrust_line, = ax3.plot([], [], 'g-', label='Compensation Thrust')
        
        time_text = ax1.text(0.05, 1.4, '', fontsize=10)
        force_text = ax1.text(0.05, 1.3, '', fontsize=10)
        
        #titles and labels
        ax1.set_title('Drone Positions and Forces')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Z Position (m)')
        ax1.grid(True)
        
        ax2.set_title('Drone Height vs Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Z Position (m)')
        ax2.grid(True)
        ax2.legend(loc='upper right')
        
        ax3.set_title('Forces and Compensation')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Force/Thrust (N)')
        ax3.grid(True)
        ax3.legend(loc='upper right')
        
        fig.suptitle(f'{model_type.title()}-based Downwash Simulation (Strength={strength}, Offset={offset}m)', 
                    fontsize=14)
        
        def update_arrow(arrow, x, y, dx, dy):
            arrow.remove()
            return ax1.arrow(x, y, dx, dy, head_width=0.03, head_length=0.05, 
                           fc=arrow.get_facecolor(), ec=arrow.get_edgecolor(), 
                           alpha=arrow.get_alpha())
        
        def init():
            if has_baseline:
                baseline_line.set_data([], [])
            comp_line.set_data([], [])
            no_comp_line.set_data([], [])
            downwash_line.set_data([], [])
            comp_thrust_line.set_data([], [])
            time_text.set_text('')
            force_text.set_text('')
            return_list = [upper_drone, lower_drone_comp, lower_drone_no_comp, 
                          comp_line, no_comp_line, downwash_line, comp_thrust_line,
                          time_text, force_text]
            if has_baseline:
                return_list.append(baseline_line)
                return_list.append(lower_drone_baseline)
            return return_list
        
        def animate(frame_idx):
            i = min(len(times) - 1, frame_idx * 5)
            
            lower_drone_comp.center = (0, lower_comp_positions[i, 2])
            lower_drone_no_comp.center = (0.2, lower_no_comp_positions[i, 2])
            if has_baseline:
                lower_drone_baseline.center = (-0.2, lower_baseline_positions[i, 2])
            
            time_range = times[:i+1]
            comp_z = lower_comp_positions[:i+1, 2]
            no_comp_z = lower_no_comp_positions[:i+1, 2]
            
            comp_line.set_data(time_range, comp_z)
            no_comp_line.set_data(time_range, no_comp_z)
            if has_baseline:
                baseline_z = lower_baseline_positions[:i+1, 2]
                baseline_line.set_data(time_range, baseline_z)
            
            downwash_line.set_data(time_range, -downwash_forces[:i+1, 2])
            comp_thrust_line.set_data(time_range, comp_thrusts[:i+1])
            
            force_scale = 0.2
            
            nonlocal downwash_arrow
            downwash_arrow = update_arrow(
                downwash_arrow, 
                upper_pos[0], upper_pos[2],  
                0, -downwash_forces[i, 2] * force_scale  
            )
            
            nonlocal comp_arrow
            comp_arrow = update_arrow(
                comp_arrow,
                0, lower_comp_positions[i, 2],  
                0, comp_thrusts[i] * force_scale  
            )
            
            time_text.set_text(f'Time: {times[i]:.1f}s')
            force_text.set_text(f'Downwash: {downwash_forces[i, 2]:.3f}N, Comp: {comp_thrusts[i]:.3f}N')
            
            return_list = [upper_drone, lower_drone_comp, lower_drone_no_comp, 
                          downwash_arrow, comp_arrow, comp_line, no_comp_line,
                          downwash_line, comp_thrust_line, time_text, force_text]
            if has_baseline:
                return_list.append(baseline_line)
                return_list.append(lower_drone_baseline)
            return return_list
        
        frames = min(50, len(times) // 5)  
        ani = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=False, interval=100)
        
        #save as GIF
        try:
            ani.save(filename, writer=PillowWriter(fps=10))
            print(f"Animation saved to {filename}")
        except Exception as e:
            print(f"Failed to save gif: {str(e)}")
            try:
                mp4_filename = filename.replace('.gif', '.mp4')
                ani.save(mp4_filename, writer='ffmpeg', fps=10)
                print(f"Animation saved as {mp4_filename}")
            except Exception as e:
                print(f"Failed to save animation: {str(e)}")
                print("Could not save animation. Please ensure pillow installed.")
                return False
        
        plt.close(fig)
        return True
    
    except Exception as e:
        print(f"Could not create animation: {str(e)}")
        return False

def create_comparison_animation(physics_result, ml_result, filename='plots/physics_vs_ml_animation.gif'):
    #takes in physics results and ml results resulting in true if done
    try:
        if (physics_result['params']['strength'] != ml_result['params']['strength'] or
            physics_result['params']['offset'] != ml_result['params']['offset']):
            print("Warning: parameters don't match between simulations")
        
        strength = physics_result['params']['strength']
        offset = physics_result['params']['offset']
        
        times = physics_result['times']
        upper_pos = physics_result['upper_pos']
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        print(f"Creating comparison animation (strength={strength}, offset={offset})...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        #physics model drone positions
        ax1 = axes[0, 0]
        ax1.set_xlim(-1.0, 1.0)
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_title('Physics-Based Model Drone Positions')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Z Position (m)')
        ax1.grid(True)
        
        #ML model drone positions 
        ax2 = axes[0, 1]
        ax2.set_xlim(-1.0, 1.0)
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_title('ML-Based Model Drone Positions')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Z Position (m)')
        ax2.grid(True)
        
        #dw forces comparison 
        ax3 = axes[1, 0]
        ax3.set_xlim(0, times[-1])
        max_force = max(
            np.max(np.abs(physics_result['downwash_forces'][:, 2])),
            np.max(np.abs(ml_result['downwash_forces'][:, 2]))
        ) * 1.2
        ax3.set_ylim(0, max_force)
        ax3.set_title('Downwash Force Comparison')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Force Magnitude (N)')
        ax3.grid(True)
        
        # drone height comparison
        ax4 = axes[1, 1]
        ax4.set_xlim(0, times[-1])
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_title('Drone Height Comparison')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Z Position (m)')
        ax4.grid(True)
        
        for ax in [ax1, ax2]:
            ax.axhline(y=0.0, color='k', linestyle=':', alpha=0.5)
        
        ax4.axhline(y=0.0, color='k', linestyle=':', alpha=0.5, label='Target')
        
        has_baseline = 'lower_baseline_positions' in physics_result and 'lower_baseline_positions' in ml_result
        
        #initialize drone circles in physics model and add baseline if available 
        upper_drone_phys = Circle((upper_pos[0], upper_pos[2]), 0.05, color='blue')
        lower_drone_comp_phys = Circle((0, physics_result['lower_comp_positions'][0, 2]), 0.05, color='green')
        lower_drone_no_comp_phys = Circle((0.2, physics_result['lower_no_comp_positions'][0, 2]), 0.05, color='red')
        ax1.add_patch(upper_drone_phys)
        ax1.add_patch(lower_drone_comp_phys)
        ax1.add_patch(lower_drone_no_comp_phys)
        if has_baseline:
            lower_drone_baseline_phys = Circle((-0.2, physics_result['lower_baseline_positions'][0, 2]), 0.05, color='black')
            ax1.add_patch(lower_drone_baseline_phys)
        
        #initialize drone circles in ml model and add baseline if available 
        upper_drone_ml = Circle((upper_pos[0], upper_pos[2]), 0.05, color='blue')
        lower_drone_comp_ml = Circle((0, ml_result['lower_comp_positions'][0, 2]), 0.05, color='green')
        lower_drone_no_comp_ml = Circle((0.2, ml_result['lower_no_comp_positions'][0, 2]), 0.05, color='red')
        ax2.add_patch(upper_drone_ml)
        ax2.add_patch(lower_drone_comp_ml)
        ax2.add_patch(lower_drone_no_comp_ml)
        if has_baseline:
            lower_drone_baseline_ml = Circle((-0.2, ml_result['lower_baseline_positions'][0, 2]), 0.05, color='black')
            ax2.add_patch(lower_drone_baseline_ml)
        
        downwash_arrow_phys = ax1.arrow(0, 0, 0, 0, head_width=0.03, head_length=0.05, fc='r', ec='r', alpha=0.7)
        comp_arrow_phys = ax1.arrow(0, 0, 0, 0, head_width=0.03, head_length=0.05, fc='g', ec='g', alpha=0.7)
     
        downwash_arrow_ml = ax2.arrow(0, 0, 0, 0, head_width=0.03, head_length=0.05, fc='r', ec='r', alpha=0.7)
        comp_arrow_ml = ax2.arrow(0, 0, 0, 0, head_width=0.03, head_length=0.05, fc='g', ec='g', alpha=0.7)
        
        phys_force_line, = ax3.plot([], [], 'r-', label='Physics-Based')
        ml_force_line, = ax3.plot([], [], 'b-', label='ML-Based')
        ax3.legend()
    
        phys_comp_line, = ax4.plot([], [], 'g-', label='Physics with Comp')
        phys_no_comp_line, = ax4.plot([], [], 'r-', label='Physics without Comp')
        ml_comp_line, = ax4.plot([], [], 'g--', label='ML with Comp')
        ml_no_comp_line, = ax4.plot([], [], 'r--', label='ML without Comp')
        if has_baseline:
            phys_baseline_line, = ax4.plot([], [], 'k-', label='Physics Baseline')
            ml_baseline_line, = ax4.plot([], [], 'k--', label='ML Baseline')
        ax4.legend()
        
        time_text = fig.text(0.5, 0.95, '', fontsize=12, ha='center')
        
        fig.suptitle(f'Physics vs ML Downwash Model comparison (Strength={strength}, Offset={offset}m)', 
                     fontsize=14)
        
        def update_arrow(ax, arrow, x, y, dx, dy):
            arrow.remove()
            return ax.arrow(x, y, dx, dy, head_width=0.03, head_length=0.05, 
                          fc=arrow.get_facecolor(), ec=arrow.get_edgecolor(), 
                          alpha=arrow.get_alpha())
        
        def init():
            phys_force_line.set_data([], [])
            ml_force_line.set_data([], [])
            phys_comp_line.set_data([], [])
            phys_no_comp_line.set_data([], [])
            ml_comp_line.set_data([], [])
            ml_no_comp_line.set_data([], [])
            time_text.set_text('')
            
            return_list = [upper_drone_phys, lower_drone_comp_phys, lower_drone_no_comp_phys,
                          upper_drone_ml, lower_drone_comp_ml, lower_drone_no_comp_ml,
                          phys_force_line, ml_force_line, phys_comp_line, phys_no_comp_line,
                          ml_comp_line, ml_no_comp_line, time_text]
            
            if has_baseline:
                phys_baseline_line.set_data([], [])
                ml_baseline_line.set_data([], [])
                return_list.extend([lower_drone_baseline_phys, lower_drone_baseline_ml,
                                   phys_baseline_line, ml_baseline_line])
            
            return return_list
        
        def animate(frame_idx):
            i = min(len(times) - 1, frame_idx * 5)
            
            #update drone positions for physics based and ML models 
            lower_drone_comp_phys.center = (0, physics_result['lower_comp_positions'][i, 2])
            lower_drone_no_comp_phys.center = (0.2, physics_result['lower_no_comp_positions'][i, 2])
            if has_baseline:
                lower_drone_baseline_phys.center = (-0.2, physics_result['lower_baseline_positions'][i, 2])

            lower_drone_comp_ml.center = (0, ml_result['lower_comp_positions'][i, 2])
            lower_drone_no_comp_ml.center = (0.2, ml_result['lower_no_comp_positions'][i, 2])
            if has_baseline:
                lower_drone_baseline_ml.center = (-0.2, ml_result['lower_baseline_positions'][i, 2])
            
            force_scale = 0.2
            nonlocal downwash_arrow_phys, comp_arrow_phys
            downwash_arrow_phys = update_arrow(
                ax1, downwash_arrow_phys, 
                upper_pos[0], upper_pos[2],  
                0, -physics_result['downwash_forces'][i, 2] * force_scale  
            )   
            comp_arrow_phys = update_arrow(
                ax1, comp_arrow_phys,
                0, physics_result['lower_comp_positions'][i, 2],  
                0, physics_result['comp_thrusts'][i] * force_scale  
            )
            
            nonlocal downwash_arrow_ml, comp_arrow_ml
            downwash_arrow_ml = update_arrow(
                ax2, downwash_arrow_ml, 
                upper_pos[0], upper_pos[2],  
                0, -ml_result['downwash_forces'][i, 2] * force_scale  
            )
            
            comp_arrow_ml = update_arrow(
                ax2, comp_arrow_ml,
                0, ml_result['lower_comp_positions'][i, 2],  
                0, ml_result['comp_thrusts'][i] * force_scale  
            )
            
            time_range = times[:i+1]
            
            phys_force_line.set_data(time_range, -physics_result['downwash_forces'][:i+1, 2])
            ml_force_line.set_data(time_range, -ml_result['downwash_forces'][:i+1, 2])
            
            phys_comp_line.set_data(time_range, physics_result['lower_comp_positions'][:i+1, 2])
            phys_no_comp_line.set_data(time_range, physics_result['lower_no_comp_positions'][:i+1, 2])
            ml_comp_line.set_data(time_range, ml_result['lower_comp_positions'][:i+1, 2])
            ml_no_comp_line.set_data(time_range, ml_result['lower_no_comp_positions'][:i+1, 2])
            
            if has_baseline:
                phys_baseline_line.set_data(time_range, physics_result['lower_baseline_positions'][:i+1, 2])
                ml_baseline_line.set_data(time_range, ml_result['lower_baseline_positions'][:i+1, 2])
            
            time_text.set_text(f'Time: {times[i]:.1f}s')
            
            return_list = [upper_drone_phys, lower_drone_comp_phys, lower_drone_no_comp_phys,
                          upper_drone_ml, lower_drone_comp_ml, lower_drone_no_comp_ml,
                          downwash_arrow_phys, comp_arrow_phys, downwash_arrow_ml, comp_arrow_ml,
                          phys_force_line, ml_force_line, phys_comp_line, phys_no_comp_line,
                          ml_comp_line, ml_no_comp_line, time_text]
            
            if has_baseline:
                return_list.extend([lower_drone_baseline_phys, lower_drone_baseline_ml,
                                   phys_baseline_line, ml_baseline_line])
            
            return return_list
        
        #animation creating
        frames = min(50, len(times) // 5) 
        ani = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=False, interval=100)
        
        #saving as gif
        try:
            ani.save(filename, writer=PillowWriter(fps=10))
            print(f"Comparison animation saved to {filename}")
        except Exception as e:
            print(f"Failed to save GIF: {str(e)}")
            try:
                mp4_filename = filename.replace('.gif', '.mp4')
                ani.save(mp4_filename, writer='ffmpeg', fps=10)
                print(f"Comparison animation saved as {mp4_filename}")
            except Exception as e:
                print(f"Failed to save animation: {str(e)}")
                print("Could not save animation. Please ensure Pillow or ffmpeg is installed.")
                return False
        
        plt.close(fig)
        return True
    
    except Exception as e:
        print(f"Could not create comparison animation: {str(e)}")
        return False

def create_metrics_visualization(results, filename='plots/performance_metrics.png'):
    try:
        strengths = []
        offsets = []
        downwash_effects = []
        compensation_improvements = []
        model_types = []
        
        for model_type, model_results in results.items():
            for result in model_results:
                strengths.append(result['params']['strength'])
                offsets.append(result['params']['offset'])
                downwash_effects.append(result['metrics']['downwash_effect'])
                compensation_improvements.append(result['metrics']['compensation_improvement'])
                model_types.append(model_type)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        for model in ['physics', 'ml']:
            mask = np.array(model_types) == model
            plt.scatter(np.array(strengths)[mask], np.array(downwash_effects)[mask], 
                      alpha=0.7, label=f'{model.title()} Model')
        
        plt.xlabel('Downwash Strength')
        plt.ylabel('Downwash Effect (%)')
        plt.title('Downwash Effect vs Strength')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        for model in ['physics', 'ml']:
            mask = np.array(model_types) == model
            plt.scatter(np.array(strengths)[mask], np.array(compensation_improvements)[mask], 
                      alpha=0.7, label=f'{model.title()} Model')
        
        plt.xlabel('Downwash Strength')
        plt.ylabel('Compensation Improvement (%)')
        plt.title('Compensation Improvement vs Strength')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        for model in ['physics', 'ml']:
            mask = np.array(model_types) == model
            plt.scatter(np.array(offsets)[mask], np.array(downwash_effects)[mask], 
                      alpha=0.7, label=f'{model.title()} Model')
        
        plt.xlabel('Horizontal Offset (m)')
        plt.ylabel('Downwash Effect (%)')
        plt.title('Downwash Effect vs Offset')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        for model in ['physics', 'ml']:
            mask = np.array(model_types) == model
            plt.scatter(np.array(offsets)[mask], np.array(compensation_improvements)[mask], 
                      alpha=0.7, label=f'{model.title()} Model')
        
        plt.xlabel('Horizontal Offset (m)')
        plt.ylabel('Compensation Improvement (%)')
        plt.title('Compensation Improvement vs Offset')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename)
        
        print(f"Performance metrics visualization saved to {filename}")
        return True
    
    except Exception as e:
        print(f"Could not create metrics visualization: {str(e)}")
        return False

if __name__ == "__main__":
    from physics_sim import run_basic_simulation
    
    print("Running basic simulation to test visualization...")
    result = run_basic_simulation()
    
    print("Creating test animation...")
    create_animation(result)