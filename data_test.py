import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from data import get_dataset, arrange_data
# a=np.loadtxt('./data/X coordinate, vehicle origin.txt')
# x=[3]
# c=x*a[0:6000]
# b=len(c)
# print(a,c,b)
# xs = []
# xs_force = []
# for i in range(2):
#     p = torch.randn(45)
#     q = torch.randn(45)
#     u = torch.randn(45)
#     xs.append(np.stack((q, p, u), axis=1))
#
# xs_force.append(np.stack(xs, axis=1))
px = np.loadtxt('./data/X coordinate, vehicle origin.txt')
py = np.loadtxt('./data/Y coordinate, vehicle origin.txt')
phi = np.loadtxt('./data/Yaw, vehicle.txt')
m = np.loadtxt('./data/sprung_mass.txt')
i = np.loadtxt('./data/yaw_inertia.txt')
vx = np.loadtxt('./data/Longitudinal speed, vehicle.txt')
vy = np.loadtxt('./data/Lateral speed, vehicle.txt')
w = np.loadtxt('./data/yaw_rate.txt')
l_delta = np.loadtxt('./data/left_steer_angle_about_kingpin.txt')
r_delta = np.loadtxt('./data/right_steer_angle_about_kingpin.txt')
tor = np.loadtxt('./data/input torque.txt')
delta = []
vy = np.array(vy)
# vx = np.sqrt(np.power(80, 2)-np.power(vy, 2))
print(vx)
# plt.subplot(2, 1, 1)
# plt.plot(vx[1500:4000])
# plt.subplot(2, 1, 2)
# plt.plot(vy[1500:4000])
plt.show()
Ix = m * vx
Iy = m * vy
Iw = i * w
# Ix = np.array(Ix)
# Ixmax = np.max(Ix)
# Ixmin = np.min(Ix)
# Ix_scale = (Ix - Ixmin) / (Ixmax - Ixmin)
#
# Iy = np.array(Iy)
# Iymax = np.max(Iy)
# Iymin = np.min(Iy)
# Iy_scale = (Iy - Iymin) / (Iymax - Iymin)
#
# Iw = np.array(Iw)
# Iwmax = np.max(Iw)
# Iwmin = np.min(Iw)
# Iw_scale = (Iw- Iwmin) / (Iwmax - Iwmin)
#
# px = np.array(px)
# pxmax = np.max(px)
# pxmin = np.min(px)
# px_scale = (px - pxmin) / (pxmax - pxmin)
#
# py = np.array(py)
# pymax = np.max(py)
# pymin = np.min(py)
# py_scale = (py - pymin) / (pymax - pymin)
#
# phi = np.array(phi)
# phimax = np.max(phi)
# phimin = np.min(phi)
# phi_scale = (phi - phimin) / (phimax - phimin)
#
# u = np.array(tor)
# umax = np.max(u)
# umin = np.min(u)
# u_scale = (u - umin) / (umax - umin)
def data_normalize(data):
    mu = np.mean(data)
    std =  np.std(data, ddof=1)
    return (data - mu)/std

Ix = np.array(Ix[1500:])
Ix_scale = data_normalize(Ix)
# Ix = np.ones(2500)

Iy = np.array(Iy[1500:])
Iy_scale = data_normalize(Iy)

Iw = np.array(Iw[1500:])
Iw_scale = data_normalize(Iw)

px = np.array(px[1500:])
px_scale = data_normalize(px)

py = np.array(py[1500:])
py_scale = data_normalize(py)

phi = np.array(phi[1500:])
phi_scale = data_normalize(phi)

u = np.array(tor[1500:])
u_scale = data_normalize(u)

p = [Ix_scale[1500:5000], Iy_scale[1500:5000], Iw_scale[1500:5000]]
q = [px_scale[1500:5000], py_scale[1500:5000], phi_scale[1500:5000]]


plt.subplot(2, 3, 1)
plt.plot(Ix_scale[1500:5000])
plt.subplot(2, 3, 2)
plt.plot(Iy_scale[1500:5000])
plt.subplot(2, 3, 3)
plt.plot(Iw_scale[1500:5000])
plt.subplot(2, 3, 4)
plt.plot(px_scale[1500:5000])
plt.subplot(2, 3, 5)
plt.plot(py_scale[1500:5000])
plt.subplot(2, 3, 6)
plt.plot(phi_scale[1500:5000])




p = [Ix_scale, Iy_scale, Iw_scale]
p=np.array(p)
print(p.shape,u.shape)

# plt.plot(Ix)
# plt.show()
for i in range(len(l_delta)):
    delta.append((l_delta[i]+r_delta[i])/2)
# p = [Ix, Iy[1500:4000], Iw[1500:4000]]
# q = [px[1500:4000], py[1500:4000], phi[1500:4000]]
q = [px_scale, py_scale, phi_scale]
p = np.array(p)
q = np.array(q)
# Ixmax=np.min(q[0,:])
# print(Ixmax,q.shape)

u = delta[1500:4000]
a = []
v = []
# for j in range(6000):
#     vt = math.sqrt(vx[i] * vx[i] + vy[i] * vy[i])
#     v.append(vt)
#
# print(v)
x_temp = []
xs = []
j = 0
xs = []
ix = torch.tensor(p[0, :], requires_grad=True, dtype=torch.float32)
iy = torch.tensor(p[1, :], requires_grad=True, dtype=torch.float32)
iw = torch.tensor(p[2, :], requires_grad=True, dtype=torch.float32)
px = torch.tensor(q[0, :], requires_grad=True, dtype=torch.float32)
py = torch.tensor(q[1, :], requires_grad=True, dtype=torch.float32)
de = torch.tensor(q[2, :], requires_grad=True, dtype=torch.float32)
ixm = torch.abs(ix).mean()
iym = torch.abs(iy).mean()
iwm = torch.abs(iw).mean()
pxm = torch.abs(px).mean()
pym = torch.abs(py).mean()
dem = torch.abs(de).mean()
print(ixm, iym, iwm, pxm, pym, dem)
# weight = 10000000*torch.ones(6)
# print(weight)
for k in range(50):
    for m in range(50):
        x_temp.append([p[0][k * 50 + m], p[1][k * 50 + m], p[2][k * 50 + m], q[0][k * 50 + m], q[1][k * 50 + m],
                       q[2][k * 50 + m], u_scale[k * 50 + m]])
    xs.append(x_temp)
    x_temp = []
xs = np.array(xs)
xs = np.stack(xs, axis=1)
xs_f = np.stack(xs,axis=0)
y = []
for i in range(50):
    for j in range(50):
        y.append(xs[j, i, :])
y = np.array(y)
print(xs.shape)
plt.plot(y[:, 6])
# print(np.abs(y[:, 2]).mean())
plt.show()

# for k in range(90):
#     for m in range(50):
#         x_temp.append([p[0][k * 50 + m], p[1][k * 50 + m], p[2][k * 50 + m], q[0][k * 50 + m], q[1][k * 50 + m],
#                        q[2][k * 50 + m], u[k * 50 + m]])
#     xs.append(x_temp)
#     x_temp = []

# t_eval = np.linspace(1, 20, 20) * 0.05
# xs = torch.tensor(xs_f, dtype=torch.float32)
# print(xs.shape)
# train_xs = xs[0,:,:]
# print(train_xs.shape)
# split_ix = int(50 * 0.5)
# train_x, test_x = xs[:, :split_ix, :], xs[:, split_ix:, :]
# print(train_x.shape)

# if_train = 0
# order = 0
#
# x = train_x[7, :, :]
#
# for i in range(train_x.shape[0]):
#     if train_x[i, :, :].equal(x) == 1:
#         order = i
#         if_train = 1
#     else:
#         continue
#
# if if_train == 1:
#     for i in range(test_x.shape[0]):
#         if test_x[i, :, :].equal(x) == 1:
#             order = i
#             if_train = 0
#         else:
#             continue

# print(order, if_train,  train_x[i, :, :].equal(x), i, x.shape)
# ar = np.array([xs])
# data = {}
# data['x'] = np.array([xs])
# split_ix = int(50 * 0.5)
# split_data = {}
# split_data['x'], split_data['test_x'] = data['x'][:, :, :split_ix, :], data['x'][:, :, split_ix:, :]
# data = split_data
# data['t'] = np.linspace(1, 120, 120) * 0.05
#
# train_x, t_eval = arrange_data(data['x'], data['t'], num_points=120)
# print(split_data['x'].shape, train_xs, t_eval)

# l_dot_delta = np.loadtxt('./data/Steer rate about kingpin L1.txt')
# R_dot_delta = np.loadtxt('./data/Steer rate about kingpin R1.txt')
# delta_dot = []
# for i in range(len(l_dot_delta[:6000])):
#     delta_dot.append((l_dot_delta[i] + R_dot_delta[i]) / 2)
# u_dot = []
# u_temp = []
# for k in range(120):
#     for m in range(50):
#         u_temp.append([delta_dot[k * 50 + m]])
#     u_dot.append(u_temp)
#     u_temp = []
#
# u = torch.tensor(u_dot, dtype=torch.float32)
# split_data = {}
# split_ix = int(50 * 0.5)
# split_data['u'], split_data['test_u'] = u[:, :split_ix, :], u[:, split_ix:, :]
# print(split_data['u'].shape)
