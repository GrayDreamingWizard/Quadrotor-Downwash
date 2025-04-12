import copy
import numpy as np
import time
import os

from Simulation.system_change import system_change
from Simulation.sim_utils import sanitize_control_dic, sanitize_trajectory_dic, merge_dicts, get_metrics
from Simulation.quadrotor import instantiate_quadrotor, downwash_model
from Simulation.trajectory import instantiate_trajectory
from Simulation.control import instantiate_controller
from Simulation.data_writer import DataWriter

def simulate(t_final, desired_speed, length, model_path, data_path):
    
    print("Starting simulation with parameters:")
    print(f"t_final={t_final}, desired_speed={desired_speed}, length={length}")
    print(f"model_path={model_path}, data_path={data_path}")
    
    sim_verbose_header = '\033[94m' + "[Simulator] " + '\033[0m'  # blue color

    #expanding for two drone setup
    upper_cf, upper_state = instantiate_quadrotor(length + 1.0) #assuming z=1 for start
    lower_cf, lower_state = instantiate_quadrotor(length)
    
    #separate trajs and controllers
    upper_cf_trajectory = instantiate_trajectory(length + 1.0, t_final, desired_speed)
    lower_cf_trajectory = instantiate_trajectory(length, t_final, desired_speed)
    upper_controller = instantiate_controller("KNODE/SavedModels/add_model_exp_weighting.pth")
    lower_controller = instantiate_controller("KNODE/SavedModels/add_model_exp_weighting.pth")

    data_write_len = 0.15  # Length of data to write (in seconds)
    data_writer = DataWriter(data_write_len, data_path)
    change_dict = {'mass': [2.0, -0.015],
                   'mass2': [5.0, 0.025]}

    t_step = 1 / 500
    total_steps = int(t_final / t_step)
    time_stamps = [0]

    upper_initial_state = {k: np.array(v) for k, v in upper_state.items()}
    lower_initial_state = {k: np.array(v) for k, v in lower_state.items()}
    upper_state = [copy.deepcopy(upper_initial_state)]
    lower_state = [copy.deepcopy(lower_initial_state)]
    upper_flat = []
    lower_flat = []
    lower_control = []
    upper_control = []

    model_cnt = 1  # model count for online learning
    mass_change_t = change_dict['mass'][0]
    data_writer.set_init_t(mass_change_t)

    for cnt in range(total_steps):
        time_stamps.append(time_stamps[-1] + t_step)
        if cnt % (0.5 / t_step) == 0:
            print(f"Step {cnt}/{total_steps}, time={time_stamps[-1]:.3f}s")
            print(f"Upper drone position: {upper_state[-1]['x']}")
            print(f"Lower drone position: {lower_state[-1]['x']}")
            print(sim_verbose_header + "Simulating {:.2f} sec".format(time_stamps[-1]))

        upper_flat.append(sanitize_trajectory_dic(upper_cf_trajectory.update(time_stamps[-1])))
        lower_flat.append(sanitize_trajectory_dic(lower_cf_trajectory.update(time_stamps[-1])))

        if cnt % 25 == 0:
            upper_ctrl_update = upper_controller.update(upper_state[-1], upper_flat[-1])
            lower_ctrl_update = lower_controller.update(lower_state[-1], lower_flat[-1])
            online_model_path = model_path + "online_model" + str(model_cnt) + ".pth"
            if cnt != 0:
                ret = lower_controller.update_model(online_model_path)
                if ret == 1:
                    model_cnt += 1
        
        #use latest controller updates
        if cnt % 25 != 0:
            upper_ctrl_update = upper_controller[-1]
            lower_ctrl_update = lower_controller[-1]
        else:
            upper_ctrl_update = {'cmd_thrust':np.zeros(1), 'cmd_moment':np.zeros(3)}
            lower_ctrl_update = {'cmd_thrust':np.zeros(1), 'cmd_moment':np.zeros(3)}

        upper_control.append(sanitize_control_dic(upper_ctrl_update))
        lower_control.append(sanitize_control_dic(lower_ctrl_update))
            
        upper_state.append(upper_cf.update(upper_state[-1], upper_control[-1], t_step))
        #lower cf gets updated with downwash 
        lower_state.append(lower_cf.update(
                        lower_state[-1],
                        lower_control[-1],
                        t_step,
                        upper_cf_state = upper_state[-1],
                        params={'magnitude':0.4, 'width':0.5, 'lateral_coeff':0.1},
                        downwash_model_fn=downwash_model
                        ))

        system_change(upper_cf, cnt * t_step, change_dict)
        system_change(lower_cf, cnt * t_step, change_dict)

        data_for_online = np.concatenate([np.hstack(list(lower_state[-1].values())), lower_control[-1]['cmd_thrust'], lower_control[-1]['cmd_moment'],], 0)
        # data writing
        if cnt * t_step > mass_change_t:
            if data_writer is not None:
                write_ret = data_writer.subscribe_data(data_for_online, cnt * t_step)

            if write_ret == 1:
                while not os.path.exists(online_model_path):  # if the node model is not ready, skip updating
                    time.sleep(3)

    # save the file with name online_end.npy to signal the end of simulation
    with open(data_path + 'online_end.npy', 'wb') as f:
        np.save(f, np.array([0]))

    #printing metrics
    # Results and metrics
    print("upper cf metrics:")
    get_metrics(merge_dicts(upper_state[:-1]), merge_dicts(upper_flat))
    print("lower cf metrics:")
    get_metrics(merge_dicts(lower_state[:-1]), merge_dicts(lower_flat))


