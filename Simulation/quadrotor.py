import numpy as np
import scipy.integrate
from Simulation.crazyflie_params import quad_params
from Simulation.sim_utils import quat_dot, rotate_k, hat_map, pack_state, unpack_state
# from KNODE import downwash_model

def downwash_model(upper_cf_state, lower_cf_state, params):
    #args state of uppper, lower, dw params
    #return: external foce/disturbance on the lower cf

    rel_pos = upper_cf_state['x'] - lower_cf_state['x']
    horz_dist = np.linalg.norm(rel_pos[:2])
    vert_dist = max(0.1, rel_pos[2])
    force_mag = params['strength']*np.exp(-horz_dist**2 / params['width']**2)/vert_dist
    force = np.array([params['lateral_coeff']*rel_pos[0], params['lateral_coeff']*rel_pos[1], -force_mag])
    return force


class Quadrotor(object):
    def __init__(self):
        self.mass = quad_params['mass']
        self.inertia = np.diag(np.array([quad_params['Ixx'], quad_params['Iyy'], quad_params['Izz']]))
        self.inv_inertia = np.linalg.inv(self.inertia)

    def update(self, state, control, t_step, upper_cf_state=None, params = None, downwash_model = None):
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, control, upper_cf_state, params, downwash_model)

        sol = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), pack_state(state), first_step=t_step)
        state = unpack_state(sol['y'][:, -1])
        return state

    def _s_dot_fn(self, t, s, control, upper_cf_state=None, params=None, downwash_model_fn=None):
        s_dot = np.zeros((13,))
        s_dot[0:3] = s[3:6]

        #base dynamics edit
        s_dot[3:6] = np.array([0, 0, -9.81]) + (control['cmd_thrust'] * rotate_k(s[6:10]))/self.mass
        if upper_cf_state is not None and params is not None and downwash_model_fn is not None:
            lower_cf_state = unpack_state(s)
            force = downwash_model_fn(upper_cf_state, lower_cf_state, params)
            s_dot[3:6] += force/self.mass
        s_dot[6:10] = quat_dot(s[6:10], s[10:13])
        s_dot[10:13] = self.inv_inertia @ (control['cmd_moment'] - hat_map(s[10:13]) @ (self.inertia @ s[10:13]))

        return s_dot


def instantiate_quadrotor(length):
    quadrotor       = Quadrotor()
    initial_state = {'x': np.array([length, 0, 0]),
                     'v': np.array([0, 0, 0]),
                     'q': np.array([0, 0., 0., 1.]),
                     'w': np.zeros(3, )}

    return quadrotor, initial_state
