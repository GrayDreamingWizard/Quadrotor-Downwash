#!/usr/bin/env python3
#physics based arch for prediction of dw forces

import numpy as np
import matplotlib.pyplot as plt
import os

class Drone:
    #drone dynamics model from KNODE github repo
    def __init__(self, mass=0.03, init_pos=np.zeros(3), init_vel=np.zeros(3)):
        self.mass = mass
        #poses and velocities in x,y and z
        self.pos = init_pos.copy()  
        self.vel = init_vel.copy()  
        self.g = 9.81 
    
    def update(self, thrust, ext_force=np.zeros(3), dt=0.01):
        #drone state w/ thrust,e xt forces and dt with total acc
        acc = np.array([0, 0, thrust/self.mass - self.g]) + ext_force/self.mass
        
        #euler integration for velocity and position update
        self.vel += acc * dt
        self.pos += self.vel * dt
        
        return self.pos.copy(), self.vel.copy()

class PhysicsDownwashModel:
    #physics based model
    def __init__(self, strength=0.3, width=0.5, lateral_coeff=0.05):
        self.strength = strength
        self.width = width
        self.lateral_coeff = lateral_coeff
    
    def calculate_force(self, upper_pos, upper_vel, lower_pos, lower_vel):
        #returns dw force in 3d: [fx, fy, fz], taking inputs upper & lower pos, upper & lower vel, and params
        # relative pos calculation
        rel_pos = upper_pos - lower_pos
        horiz_dist = np.linalg.norm(rel_pos[:2])
        
        #check to ensure lower below upper to apply dw
        if rel_pos[2] <= 0:
            return np.zeros(3)
        
        vert_dist = max(0.1, rel_pos[2])
        
        #simple model for dw:
        # Gaussian distribution horizontally and inverse-squared distance vertically
        
        magnitude = self.strength * np.exp(-horiz_dist**2 / (2 * self.width**2)) / (vert_dist * vert_dist)
        
        #upper velocity faster results in stronger dw
        vel_factor = 1.0 + 0.1 * abs(upper_vel[2])
        magnitude *= vel_factor
        
        #lateral force calc
        lateral_force = np.zeros(2)
        if horiz_dist > 0.001:  # Avoid division by zero
            lateral_dir = rel_pos[:2] / horiz_dist
            lateral_force = self.lateral_coeff * magnitude * lateral_dir
        
        #vertical and lateral forces combined
        downwash_force = np.array([lateral_force[0], lateral_force[1], -magnitude])
        
        max_force = 0.8 * 9.81 * 0.03 
        if np.linalg.norm(downwash_force) > max_force:
            downwash_force = downwash_force / np.linalg.norm(downwash_force) * max_force
        
        return downwash_force

def pid_controller(target_pos, current_pos, current_vel, dt, Kp = 20.0, Kd = 10.0, Ki = 0.5):
    """
    drone position ctrl using PID
    args: target pos, current pos and current vel
        dt and PID gains
    returns: thrust cmd
    """
    if not hasattr(pid_controller, "integral_error"):
        pid_controller.integral_error = np.zeros(3)
    
    #error calc and update integral error
    pos_error = target_pos - current_pos
    
    pid_controller.integral_error += pos_error * dt
    max_integral = 0.5  
    for i in range(3):
        if abs(pid_controller.integral_error[i]) > max_integral:
            pid_controller.integral_error[i] = np.sign(pid_controller.integral_error[i]) * max_integral
    
    #proporional, integral and derivative ctrl
    p_term = Kp * pos_error  
    i_term = Ki * pid_controller.integral_error  
    d_term = -Kd * current_vel  #negative to dampen velocity
    
    #acc cmd
    acc_cmd = p_term + i_term + d_term
    
    #acc to thrust mapping
    mass = 0.03  
    thrust_cmd = mass * (9.81 + acc_cmd[2])  #mass*(g+acc_z)
    
    return max(0, thrust_cmd) 

def run_basic_simulation(strength=0.3, offset=0.0, simulation_time=5.0):
    #basic physics-based simulation with fixed upper drone height and fixed dw params
    print(f"running physics simulation (strength={strength}, offset={offset})...")
    
    #sim params
    dt = 0.01  #dt (s)
    steps = int(simulation_time / dt)
    times = np.linspace(0, simulation_time, steps)
    
    #upper drone at fixed height with specified horz offset
    upper_pos = np.array([offset, 0, 1.0])
    upper_vel = np.zeros(3)
    
    #lower drone initialization, no dw baseline init
    lower_drone_with_comp = Drone(init_pos=np.array([0, 0, 0.0]))
    lower_drone_without_comp = Drone(init_pos=np.array([0, 0, 0.0]))
    lower_drone_baseline = Drone(init_pos=np.array([0, 0, 0.0]))
    
    downwash_model = PhysicsDownwashModel(strength=strength)
    
    lower_comp_positions = np.zeros((steps, 3))
    lower_comp_velocities = np.zeros((steps, 3))
    lower_no_comp_positions = np.zeros((steps, 3))
    lower_no_comp_velocities = np.zeros((steps, 3))
    lower_baseline_positions = np.zeros((steps, 3))
    downwash_forces = np.zeros((steps, 3))
    comp_thrusts = np.zeros(steps)
    no_comp_thrusts = np.zeros(steps)
    
    lower_comp_positions[0] = lower_drone_with_comp.pos
    lower_no_comp_positions[0] = lower_drone_without_comp.pos
    lower_baseline_positions[0] = lower_drone_baseline.pos
    
    # lower drone target at z=0
    lower_target = np.array([0, 0, 0.0])
    
    # PID controller state reset
    pid_controller.integral_error = np.zeros(3)
    
    #simulation loop
    for i in range(1, steps):
        #dw force affecting lower drone
        downwash_force = downwash_model.calculate_force(
            upper_pos, upper_vel, lower_drone_with_comp.pos, lower_drone_with_comp.vel
        )
        downwash_forces[i] = downwash_force
        
        #full compensation thrust calculation
        comp_thrust = -downwash_force[2] * lower_drone_with_comp.mass * 10.0
        comp_thrusts[i] = comp_thrust 
        
        #lower drone ctrl with compensation
        lower_comp_base_thrust = pid_controller(
            lower_target, lower_drone_with_comp.pos, lower_drone_with_comp.vel, dt
        )
        lower_comp_thrust = lower_comp_base_thrust + comp_thrust
        comp_thrusts[i] = comp_thrust
        
        # lower drone ctrl w/o compensation
        lower_no_comp_thrust = pid_controller(
            lower_target, lower_drone_without_comp.pos, lower_drone_without_comp.vel, dt
        )
        no_comp_thrusts[i] = lower_no_comp_thrust
        
        #lower drone ctrl with no dw (baseline)
        lower_baseline_thrust = pid_controller(
            lower_target, lower_drone_baseline.pos, lower_drone_baseline.vel, dt
        )
        
        #update state
        lower_comp_positions[i], lower_comp_velocities[i] = lower_drone_with_comp.update(
            lower_comp_thrust, downwash_force, dt
        )
        
        #update state for case w/o compensation
        lower_no_comp_positions[i], lower_no_comp_velocities[i] = lower_drone_without_comp.update(
            lower_no_comp_thrust, downwash_force, dt
        )
        
        #update state for case w/o dw
        lower_drone_baseline.update(lower_baseline_thrust, np.zeros(3), dt)
        lower_baseline_positions[i] = lower_drone_baseline.pos

        if i % (steps // 5) == 0:
            print(f"DEBUG: Base thrust: {lower_comp_base_thrust:.6f}N, Comp thrust: {comp_thrust:.6f}N, Total: {lower_comp_thrust:.6f}N")
    
    #metric calculatins: RMSE and improvement with comp
    baseline_rmse = np.sqrt(np.mean((lower_baseline_positions[:, 2] - lower_target[2])**2))
    lower_comp_rmse = np.sqrt(np.mean((lower_comp_positions[:, 2] - lower_target[2])**2))
    lower_no_comp_rmse = np.sqrt(np.mean((lower_no_comp_positions[:, 2] - lower_target[2])**2))
    downwash_effect = (lower_no_comp_rmse - baseline_rmse) / baseline_rmse * 100
    compensation_improvement = (lower_no_comp_rmse - lower_comp_rmse) / lower_no_comp_rmse * 100
    
    print("sim complete!")
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
            'offset': offset
        }
    }

if __name__ == "__main__":
    result = run_basic_simulation(strength=0.3, offset=0.0)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(result['times'], result['lower_baseline_positions'][:, 2], 'b-', label='baseline (No Downwash)')
    plt.plot(result['times'], result['lower_comp_positions'][:, 2], 'g-', label='With Compensation')
    plt.plot(result['times'], result['lower_no_comp_positions'][:, 2], 'r-', label='Without Compensation')
    plt.axhline(y=0.0, color='k', linestyle=':', alpha=0.5, label='Target')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.title('Drone Position with and without Downwash Compensation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(result['times'], -result['downwash_forces'][:, 2], 'r-', label='Downwash Force')
    plt.plot(result['times'], result['comp_thrusts'], 'g-', label='Compensation Thrust')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Downwash Forces and Compensation')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    plt.tight_layout()
    plt.savefig('plots/basic_simulation.png')
    plt.show()