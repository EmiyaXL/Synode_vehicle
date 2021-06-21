# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# code structure follows the style of HNN by Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import autograd
import autograd.numpy as np
import numpy as np
__all__ = [np]

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from utils import to_pickle, from_pickle
import gym
import torch
import myenv

def hamiltonian_fn(coords):
    q, p = np.split(coords,2)
    # pendulum hamiltonian conosistent with openAI gym Pendulum-v0
    H = 5*(1-np.cos(q)) + 1.5 * p**2 
    return H

def dynamics_fn(t, coords, u=0):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords,2)
    S = np.concatenate([dpdt, -dqdt + u], axis=-1)
    return S

def get_trajectory(timesteps=20, radius=None, y0=None, noise_std=0.1, u=0.0, rad=False, **kwargs):
    t_eval = np.linspace(1, timesteps, timesteps) * 0.05
    t_span = [0.05, timesteps*0.05]

    # get initial state
    if rad:
        if y0 is None:
            y0 = np.random.rand(2)*2.-1
        if radius is None:
            radius = np.random.rand() + 1.3 # sample a range of radii
        y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius
    else:
        if y0 is None:
            y0 = np.random.rand(2) * 3 * np.pi - np.pi

    spring_ivp = solve_ivp(lambda t, y: dynamics_fn(t, y, u), t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]

    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std

    return q, p, t_eval

def data_normalize(data):
    mu = np.mean(data)
    std =  np.std(data, ddof=1)
    return (data - mu)/std

def read_data():
    px = np.loadtxt('./data/car_sim mu=0.65 v=50/X coordinate, vehicle origin.txt')
    py = np.loadtxt('./data/car_sim mu=0.65 v=50/Y coordinate, vehicle origin.txt')
    phi = np.loadtxt('./data/car_sim mu=0.65 v=50/Yaw, vehicle.txt')
    m = np.loadtxt('./data/car_sim mu=0.65 v=50/sprung_mass.txt')
    i = np.loadtxt('./data/car_sim mu=0.65 v=50/yaw_inertia.txt')
    vx = np.loadtxt('./data/car_sim mu=0.65 v=50/Longitudinal speed, vehicle.txt')
    vy = np.loadtxt('./data/car_sim mu=0.65 v=50/Lateral speed, vehicle.txt')
    w = np.loadtxt('./data/car_sim mu=0.65 v=50/yaw_rate.txt')
    tor = np.loadtxt('./data/car_sim mu=0.65 v=50/input torque.txt')

    Ix = m * vx
    Iy = m * vy
    Iw = i * w

    # data scale
    Ix = np.array(Ix)
    Ix_scale = data_normalize(Ix)
    # Ix = np.ones(2500)

    Iy = np.array(Iy)
    Iy_scale = data_normalize(Iy)

    Iw = np.array(Iw)
    Iw_scale = data_normalize(Iw)

    px = np.array(px)
    px_scale = data_normalize(px)

    py = np.array(py)
    py_scale = data_normalize(py)

    phi = np.array(phi)
    phi_scale = data_normalize(phi)

    u = np.array(tor)
    u_tmp = data_normalize(u)
    u_scale = u_tmp[2000:6000]

    p = [Ix_scale[2000:6000], Iy_scale[2000:6000], Iw_scale[2000:6000]]
    q = [px_scale[2000:6000], py_scale[2000:6000], phi_scale[2000:6000]]
    # p = [Ix[2000:6000], Iy[2000:6000], Iw[2000:6000]]
    # q = [px[2000:6000], py[2000:6000], phi[2000:6000]]
    # p = [Iy_scale, Iw_scale]
    # q = [py_scale, phi_scale]


    x_temp = []
    xs = []
    for k in range(50):
        for m in range(80):
            # x_temp.append([p[0][k * 90 + m], p[1][k * 90 + m], p[2][k * 90 + m], q[0][k * 90 + m], q[1][k * 90 + m],
            #                q[2][k * 90 + m], u[k * 90 + m]])
            x_temp.append([q[0][k * 80 + m], q[1][k * 80 + m], q[2][k * 80 + m], p[0][k * 80 + m], p[1][k * 80 + m],
                           p[2][k * 80 + m], tor[k * 80 + m]])
            # x_temp.append([q[0][k * 50 + m], q[1][k * 50 + m], p[0][k * 50 + m], p[1][k * 50 + m], u_scale[k * 50 + m]])
        xs.append(x_temp)
        x_temp = []
    xs_f = np.stack(xs, axis=1)

    return xs_f

def get_dataset(seed=0, samples=50, test_split=0.5, gym=False, save_dir=None, us=[0], rad=False, timesteps=20, **kwargs):
    # data = {}
    #
    # xs_force = []
    # for u in us:
    #     xs = []
    #     np.random.seed(seed)
    #     for _ in range(samples):
    #         q, p, t = get_trajectory(noise_std=0.0, u=u, rad=rad, timesteps=timesteps, **kwargs)
    #         xs.append(np.stack((q, p, np.ones_like(q)*u), axis=1)) # (45, 3) last dimension is u
    #     xs_force.append(np.stack(xs, axis=1)) # fit Neural ODE format (45, 50, 3)
    #
    # data['x'] = np.stack(xs_force, axis=0) # (3, 45, 50, 3)
    #
    # # make a train/test split
    # split_ix = int(samples * test_split)
    # split_data = {}
    # split_data['x'], split_data['test_x'] = data['x'][:,:,:split_ix,:], data['x'][:,:,split_ix:,:]
    #
    # data = split_data
    # data['t'] = t

    ### New Part
    data = {}
    xs = read_data()
    data['x'] = np.array([xs]) # (1, 120, 50, 7)
    split_ix = int(samples * test_split)
    split_data = {}
    split_data['x'], split_data['test_x'] = data['x'][:, :, :split_ix, :], data['x'][:, :, split_ix:, :]
    data = split_data
    data['t'] = np.linspace(1, timesteps, timesteps) * 0.025
    return data


def arrange_data(x, t, num_points=2):
    '''Arrange data to feed into neural ODE in small chunks'''
    assert num_points>=2 and num_points<=len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points-1:
            x_stack.append(x[:, i:-num_points+i+1,:,:])
        else:
            x_stack.append(x[:, i:, :, :])
    x_stack = np.stack(x_stack, axis=1)
    x_stack = np.reshape(x_stack, 
                (x.shape[0], num_points, -1, x.shape[3]))
    t_eval = t[0:num_points]
    return x_stack, t_eval


def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20, u=0):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y, u) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field

def get_order(x):
    x_tmp = read_data()
    xs = torch.tensor(x_tmp, dtype=torch.float32)

    split_ix = int(50 * 0.5)
    train_x, test_x = xs[:, :split_ix, :-1], xs[:, split_ix:, :-1]
    if_train = 0
    order = 0

    for i in range(train_x.shape[0]):
        if train_x[i, :, :].equal(x) == 1:
            order = i
            if_train = 1
        else:
            continue

    if if_train == 1:
        for i in range(test_x.shape[0]):
            if test_x[i, :, :].equal(x) == 1:
                order = i
                if_train = 0
            else:
                continue

    return order, if_train

def get_dot_u():
    l_dot_delta = np.loadtxt('./data/Steer rate about kingpin L1.txt')
    R_dot_delta = np.loadtxt('./data/Steer rate about kingpin R1.txt')
    delta_dot = []
    for i in range(len(l_dot_delta[1500:6000])):
        delta_dot.append((l_dot_delta[i] + R_dot_delta[i]) / 2)
    us = []
    u_temp = []
    for k in range(90):
        for m in range(50):
            u_temp.append([delta_dot[k * 50 + m]])
        us.append(u_temp)
        u_temp = []

    split_data = {}
    data = {}
    split_ix = int(50 * 0.5)
    u_temp = torch.tensor(us, dtype=torch.float32)
    dot_u, test_dot_u = u_temp[:, :split_ix, :], u_temp[:, split_ix:, :]

    return dot_u, test_dot_u

def get_u():
    tor = np.loadtxt('./data/input torque.txt')
    torq =tor[1500:6000]
    us = []
    u_temp = []
    for k in range(90):
        for m in range(50):
            u_temp.append([torq[k * 50 + m]])
        us.append(u_temp)
        u_temp = []

    split_data = {}
    data = {}
    split_ix = int(50 * 0.5)
    u_temp = torch.tensor(us, dtype=torch.float32)
    u, test_u = u_temp[:, :split_ix, :], u_temp[:, split_ix:, :]

    return u, test_u

def get_data(seed=0, samples=50, test_split=0.5, gym=False, save_dir=None, us=[0], rad=False, timesteps=20, **kwargs):
    ### New Part
    data = {}
    xs = read_data()
    data['x'] = np.array([xs]) # (1, 120, 50, 7)
    # split_ix = int(samples * test_split)
    # split_data = {}
    # split_data['x'], split_data['test_x'] = data['x'][:, :, :split_ix, :], data['x'][:, :, split_ix:, :]
    # data = split_data
    data['t'] = np.linspace(1, timesteps, timesteps) * 0.025
    return data
