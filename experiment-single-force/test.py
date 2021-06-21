# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# code structure follows the style of HNN by Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import torch, argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP, PSD,MLP1
from symoden import SymODEN_R
from data import \
    get_dataset, arrange_data, get_data
from utils import L2_loss, to_pickle
import time


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=6, type=int, help='dimensionality of input tensor')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=7000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='pend', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_false', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=80,
                        help='number of evaluation points by the ODE solver, including the initial point')
    parser.add_argument('--structure', dest='structure', action='store_false', help='using a structured Hamiltonian')
    parser.add_argument('--rad', dest='rad', action='store_true', help='generate random data around a radius')
    parser.add_argument('--solver', default='euler', type=str, help='type of ODE Solver for Neural ODE')
    parser.set_defaults(feature=True)
    return parser.parse_args()

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# setup_seed(20)

def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def train(args):
    # import ODENet
    # from torchdiffeq import odeint
    from torchdiffeq import odeint_adjoint as odeint

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # reproducibility: set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init model and optimizer
    if args.verbose:
        print("Start training with num of points = {} and solver {}.".format(args.num_points, args.solver))

    if args.structure == False and args.baseline == True:
        nn_model = MLP(args.input_dim, 8, args.input_dim, args.nonlinearity).to(device)
        model = SymODEN_R(args.input_dim, H_net=nn_model, device=device, baseline=True)
    elif args.structure == False and args.baseline == False:
        H_net = MLP(args.input_dim, 8, 1, args.nonlinearity).to(device)
        # g_net = MLP(int(args.input_dim/2), 200, int(args.input_dim/2)).to(device)
        g_net = MLP(3, 8, 3).to(device)
        model = SymODEN_R(args.input_dim,  H_net=H_net, g_net=g_net, device=device, baseline=False)
    elif args.structure == True and args.baseline == False:
        M_net = MLP(int(args.input_dim / 2), 300, int(args.input_dim / 2))
        V_net = MLP(int(args.input_dim / 2), 50, 1).to(device)
        g_net = MLP(int(args.input_dim / 2), 200, int(args.input_dim / 2)).to(device)
        model = SymODEN_R(args.input_dim, M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=False,
                          structure=True).to(device)
    else:
        raise RuntimeError('argument *baseline* and *structure* cannot both be true')

    num_parm = get_model_parm_nums(model)
    print('model contains {} parameters'.format(num_parm))

    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    # arrange data
    us = [-2.0, -1.0, 0.0, 1.0, 2.0]
    # us = [0.0]
    data = get_data(seed=args.seed, timesteps=80,
                       save_dir=args.save_dir, rad=args.rad, us=us, samples=50)
    train_x, t_eval = arrange_data(data['x'], data['t'], num_points=args.num_points)
    # test_x, t_eval = arrange_data(data['test_x'], data['t'], num_points=args.num_points)
    train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float32).to(device)
    # test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)

    print(train_x.shape)

    # training loop
    stats = {'train_loss': [], 'test_loss': [], 'forward_time': [], 'backward_time': [], 'nfe': []}
    for step in range(args.total_steps + 1):
        train_loss = 0
        test_loss = 0
        for i in range(train_x.shape[0]):
            t = time.time()
            train_x_hat = []
            for j in range(train_x.shape[1] - 1):
                model.get_u(train_x[i, j, :, -1:])
                # print(train_x[i, j, :, :-1].shape)
                # print(tmp_x_hat[-1].shape)
                # print(train_x[i, j, :, :-1])
                # print(args.structure, args.baseline)

                if j == 0:
                    tmp_x_hat = odeint(model, train_x[i, j, :, :-1], torch.tensor([0.025 * j, 0.025 * (j + 1)]),method=args.solver)
                    train_x_hat.append(tmp_x_hat[0])
                else:
                    tmp_x_hat = odeint(model, tmp_x_hat[-1], torch.tensor([0.025 * j, 0.025 * (j + 1)]),method=args.solver)
                train_x_hat.append(tmp_x_hat[-1])
                # else:
                #     tmp_x_hat = odeint(model, tmp_x_hat[-1], torch.tensor([0, 0.025]), method=args.solver)
                #     print(tmp_x_hat.shape)

            train_x_hat = torch.stack(train_x_hat, dim=0)
            # print('train x hat :{}'.format(train_x_hat.shape))
            forward_time = time.time() - t
            # train_weight = torch.tensor([1000, 100, 1000, 1, 10000, 100], requires_grad=True, dtype=torch.float32)
            # train_loss_mini = L2_loss(train_x[i, :, :, :-1]*train_weight, train_x_hat*train_weight)
            train_loss_mini = L2_loss(train_x[i, :, :, :-1], train_x_hat)
            # train_loss_test = (train_x[i, :, :, :-1]-train_x_hat).pow(2)
            # print(train_loss_test)
            train_loss = train_loss + train_loss_mini

            t = time.time()
            train_loss_mini.backward();
            optim.step();
            optim.zero_grad()
            backward_time = time.time() - t

            # run test data
            # test_x_hat = odeint(model, test_x[i, 0, :, :-1], t_eval, method=args.solver)
            test_x_hat = []
        #     for k in range(test_x.shape[1] - 1):
        #         model.get_u(test_x[i, k, :, -1:])
        #         tmp_x_hat = odeint(model, test_x[i, 0, :, :-1], torch.tensor([0, 0.025]), method=args.solver)
        #         if k == 0:
        #             test_x_hat.append(tmp_x_hat[0])
        #         test_x_hat.append(tmp_x_hat[-1])
        #     test_x_hat = torch.stack(test_x_hat, dim=0)
        #     test_loss_mini = L2_loss(test_x[i, :, :, :-1], test_x_hat)
        #     test_loss = test_loss + test_loss_mini
        print(step, train_loss)
        # logging
        stats['train_loss'].append(train_loss.item())
        # stats['test_loss'].append(test_loss.item())
        stats['forward_time'].append(forward_time)
        stats['backward_time'].append(backward_time)
        stats['nfe'].append(model.nfe)
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, train_loss.item(), test_loss.item()))

    # calculate loss mean and std for each traj.
    train_x, t_eval = data['x'], data['t']
    # test_x, t_eval = data['test_x'], data['t']

    train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float32).to(device)
    # test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)

    train_loss = []
    test_loss = []
    for i in range(train_x.shape[0]):
        # train_x_hat = odeint(model, train_x[i, 0, :, :-1], t_eval, method=args.solver)
        # train_loss.append((train_x[i, :, :, :-1] - train_x_hat) ** 2)
        # print(train_loss)
        train_x_hat = []
        for j in range(train_x.shape[1] - 1):
            model.get_u(train_x[i, j, :, -1:])
            if j == 0:
                tmp_x_hat = odeint(model, train_x[i, j, :, :-1], torch.tensor([0, 0.025]), method=args.solver)
                train_x_hat.append(tmp_x_hat[0])
            else:
                tmp_x_hat = odeint(model, tmp_x_hat[-1], torch.tensor([0.025*j, 0.025*(j+1)]), method=args.solver)
            train_x_hat.append(tmp_x_hat[-1])
        train_x_hat = torch.stack(train_x_hat, dim=0)
        # train_weight = torch.tensor([1000, 100, 1000, 1, 10000, 100], dtype=torch.float32)
        train_loss.append((train_x[i, :, :, :-1] - train_x_hat) ** 2)
        # run test data
        # test_x_hat = odeint(model, test_x[i, 0, :, :-1], t_eval, method=args.solver)
        # test_loss.append((test_x[i, :, :, :-1] - test_x_hat) ** 2)

    train_loss = torch.cat(train_loss, dim=1)
    train_loss_per_traj = torch.sum(train_loss, dim=(0, 2))

    y = torch.empty(4000, 6)
    y_hat = torch.empty(4000, 6)
    with torch.no_grad():
        for i in range(50):
            for j in range(80):
                y[i * 80 + j, :] = train_x[0, j, i, :-1]
                y_hat[i * 80 + j, :] = train_x_hat[j, i, :]
    # y = np.array(y)
    # y_hat = np.array(y_hat)
    ts = y.shape[0]
    x = np.linspace(1, ts, ts)*0.025
    yn = np.stack([y, y_hat], axis=1)
    for l in range(6):
        plt.subplot(2, 3, l+1)
        plt.plot(yn[:, :, l])



    # test_loss = torch.cat(test_loss, dim=1)
    # test_loss_per_traj = torch.sum(test_loss, dim=(0, 2))

    # print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
    #       .format(train_loss_per_traj.mean().item(), train_loss_per_traj.std().item(),
    #               test_loss_per_traj.mean().item(), test_loss_per_traj.std().item()))
    print('Final trajectory train loss {:.4e} +/- {:.4e}'
          .format(train_loss_per_traj.mean().item(), train_loss_per_traj.std().item()))

    stats['traj_train_loss'] = train_loss_per_traj.detach().cpu().numpy()
    # stats['traj_test_loss'] = test_loss_per_traj.detach().cpu().numpy()
    plt.show()
    return model, stats


if __name__ == "__main__":
    from torchdiffeq import odeint_adjoint as odeint
    trainornot = True
    path = 'D:/ghhhub/Symplectic-ODENet/experiment-single-force/model_new.pkl'
    args = get_args()
    if trainornot:
        print(mpl.get_backend())
        model, stats = train(args)
        torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)
    else:
        device = 'cuda' if torch.cuda.is_available() else "cpu"
        H_net = MLP(args.input_dim, 50, 1, args.nonlinearity).to(device)
        # g_net = MLP(int(args.input_dim/2), 200, int(args.input_dim/2)).to(device)
        g_net = MLP(3, 50, 3).to(device)
        model = SymODEN_R(args.input_dim, H_net=H_net, g_net=g_net, device=device, baseline=False)
        model.load_state_dict(torch.load(path))
        p_q = torch.tensor([1,2,3,4,5,6], device=device, dtype=torch.float32)
        u = torch.tensor([1], device=device, dtype=torch.float32)
        p_q = p_q.reshape((1,6))
        u = u.reshape((1,1))
        model.get_u(u)
        res1 = odeint(model, p_q, torch.tensor([0, 0.025]), method=args.solver)
        res = model(0, p_q)
        print(res)
        print('='*80)
        print(res1)
    # save
    # os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    # label = '-baseline_ode' if args.baseline else '-hnn_ode'
    # struct = '-struct' if args.structure else ''
    # rad = '-rad' if args.rad else ''


    # path = '{}/{}{}{}-{}-p{}-stats{}.pkl'.format(args.save_dir, args.name, label, struct, args.solver, args.num_points,
    #                                              rad)
    # to_pickle(stats, path)


    #
