import sys
import os
import pickle
import numpy as np
from prettytable import PrettyTable

# tracking error
# import ipdb;ipdb.set_trace()
# error = (np.expand_dims(traj,-1) - np.expand_dims(ref,-1).transpose([-1,1,2,0])) # T x N x 8 x M
# error = error[:,:,:3,:] # T x N x 3 x M
# error = np.sqrt((error**2).sum(axis=2)) # T x N x M
# error = error.min(axis=-1) # T x N
# error = error.mean()
# print('tracking error:', error)

def get_distance_between_agents(X):
    # distance between agents
    # X: T x N x 8
    num_agents = X.shape[1]
    X = np.expand_dims(X[:,:,:3], -1) # T x N x 3 x 1
    error = (X - X.transpose([0,3,2,1])) # T x N x 3 x N
    error = np.sqrt((error**2).sum(axis=2)) # T x N x N
    error[:, np.eye(num_agents).astype('bool')] = np.inf # T x N x N
    error = error.min(axis=-1) # T x N
    return error
# error.min(axis=0)
# error[0,:].min()
# print(error.min())

def get_distance_between_agents_and_obs(X, obstacles):
    # distance between agents and obstacles
    # X: T x N x 8
    # obs: O x 6
    obs = np.expand_dims(np.array(obstacles).T, [0,1]) # 1 x 1 x 6 x O
    centers = (obs[:,:,:3,:] + obs[:,:,3:,:]) / 2 # 1 x 1 x 3 x O
    half_width = (obs[:,:,3:,:] - obs[:,:,:3,:]) / 2 # 1 x 1 x 3 x O
    X = np.expand_dims(X[:,:,:3], -1) # T x N x 3 x 1
    error = np.maximum(0, np.abs((X - centers)) - half_width) # T x N x 3 x O
    error = np.sqrt((error ** 2).sum(axis=2)) # T x N x O
    error = error.min(axis=-1) # T x N
    return error

def eval_one(filename):
    data = pickle.load(open(filename, 'rb'))
    traj = data['traj'] # T x N x 8
    lqr = data['lqr']
    path = data['path']
    dists = np.array(data['dists']) # T x N
    dist_obs = np.array(data['dist_obs'])
    dist_agents = np.array(data['dist_agents'])
    tracking_errors = np.array(data['tracking_errors'])

    num_agents = traj.shape[1]

    # running time
    # print('running time: %.3f s'%(traj.shape[0]*0.01))
    RT = traj.shape[0]*0.01

    # number of safe agents
    num_safe_agents = (dists.min(axis=0) > 1e-5).sum()
    # print('num_safe_agents: %d/%d = %.3f%%'%(num_safe_agents, num_agents, 100.*num_safe_agents/num_agents))
    NSA =  num_safe_agents

    # safe time / total time (only for w/o removal)
    ratio = (dists > 1e-5).sum(axis=0) / dists.shape[0]
    # print('safe time / total time: %.3f'%(ratio.mean()))
    SoT = ratio.mean()

    # tracking error
    # print('tracking error: %.3f'%np.nanmean(tracking_errors))
    TE = np.nanmean(tracking_errors)
    return SoT, NSA, RT, TE

if __name__ == '__main__':
    num_agents = [4, 8, 16, 32]
    seed = list(range(5))
    noise_level = [0, 0.1, 0.5]

    x = PrettyTable()
    x.field_names = ["Method", "N", "noise", "dis_c", "SoT", "NSA", "RT", "TE"]

    for n in noise_level:
        for a in num_agents:
            res = []
            res_removal = []
            for s in seed:
                fn = "data/run/data_a{num_agents}_n{noise_level:.1f}_s{seed}_discheck0.6.pkl".format(num_agents=a, noise_level=n, seed=s)
                if os.path.exists(fn):
                    res.append(eval_one(fn))
                fn = "data/run/data_a{num_agents}_n{noise_level:.1f}_s{seed}_removal_discheck0.6.pkl".format(num_agents=a, noise_level=n, seed=s)
                if os.path.exists(fn):
                    res_removal.append(eval_one(fn))
            if len(res) > 0:
                x.add_row(['w/o removal', a, n, 0.6] + np.array(res).mean(axis=0).tolist())
            if len(res_removal) > 0:
                x.add_row(['w/ removal', a, n, 0.6] + np.array(res_removal).mean(axis=0).tolist())
    print(x)
