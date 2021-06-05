import numpy as np
import sys
import os
import gurobi_MILP
gurobi_MILP.setM(1e6)
from gurobi_MILP import plan as plan_gurobi
import LQR
import C3M
import hashlib
import pickle

def interpolate(wp_in):
    dt = 0.01 # FIXME
    v = 3. # FIXME
    wp_out = []
    t = []
    currtent_t = 0.
    for i in range(wp_in.shape[0]-1):
        p1 = wp_in[i, :]
        p2 = wp_in[i+1, :]
        dist = np.sqrt(((p2 - p1)**2).sum())
        unit = (p2 - p1) / dist
        T = dist / v
        local_t = np.arange(0., T, dt)
        wp_out += [p1+lt*v*unit for lt in local_t]
        t += (local_t+currtent_t).tolist()
        currtent_t = t[-1] + dt
    t.append(currtent_t)
    wp_out.append(wp_in[-1,:])
    return np.array(wp_out), np.array(t)

def plot_env(x_init, x_goal, obstacles, path=None, lqr=None, c3m=None, colors=None, x_bound=None, bloat_factor=0.):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i, ob in enumerate(obstacles):
        ob = ob[1]
        ob = np.array([-ob[0], -ob[2], -ob[4], ob[1], ob[3], ob[5]])+np.array([-1,-1,-1,1,1,1])*bloat_factor
        ax.bar3d(ob[0], ob[1], ob[2], ob[3]-ob[0], ob[4]-ob[1], ob[5]-ob[2], color='red')
        ax.text(ob[0], ob[1], 100., str(i), color='k')
    ax.plot([x[0] for x in x_init], [x[1] for x in x_init], [x[2] for x in x_init], 'bo', markersize=5.)
    ax.plot([x[0] for x in x_goal], [x[1] for x in x_goal], [x[2] for x in x_goal], 'ko', markersize=5.)
    if path is not None:
        for idx in range(len(path)):
            ax.plot(path[idx][:,0], path[idx][:,1], path[idx][:,2], '-.', color=colors[idx])
    if lqr is not None:
        for idx in range(len(lqr)):
            ax.plot(lqr[idx][:,0], lqr[idx][:,1], lqr[idx][:,2], '--', color=colors[idx])
    if c3m is not None:
        for idx in range(len(c3m)):
            ax.plot(c3m[idx][:,0], c3m[idx][:,1], c3m[idx][:,2], '-', color=colors[idx])
    plt.show()

def plot_env2d(x_init, x_goal, obstacles, path=None, lqr=None, c3m=None, colors=None, x_bound=None, bloat_factor=0.):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure()
    ax = fig.gca()
    for i, ob in enumerate(obstacles):
        ob = ob[1]
        ob = np.array([-ob[0], -ob[2], -ob[4], ob[1], ob[3], ob[5]])+np.array([-1,-1,-1,1,1,1])*bloat_factor
        box = Rectangle(ob[0:2], ob[3]-ob[0], ob[4]-ob[1], facecolor='r')
        ax.add_patch(box)
        ax.text(ob[0], ob[1], str(i), color='k')
    ax.plot([x[0] for x in x_init], [x[1] for x in x_init], 'bo', markersize=5.)
    ax.plot([x[0] for x in x_goal], [x[1] for x in x_goal], 'ko', markersize=5.)
    if path is not None:
        for idx in range(len(path)):
            ax.plot(path[idx][:,0], path[idx][:,1], '-.', color=colors[idx])
    if lqr is not None:
        for idx in range(len(lqr)):
            ax.plot(lqr[idx][:,0], lqr[idx][:,1], '--', color=colors[idx])
    if c3m is not None:
        for idx in range(len(c3m)):
            ax.plot(c3m[idx][:,0], c3m[idx][:,1], '-', color=colors[idx])
    plt.show()

def negmu(x1, x2, bloat_factor, A, b, EPS=1e-2):
    # this segment is outside Ax<=b (bloated)
    b = b.reshape(-1)
    num_edges = len(b)
    disjunctions = []
    for e in range(num_edges):
        a = A[e,:]
        conjunctions = []
        conjunctions.append(sum([a[k]*x1[k] for k in range(len(x1))]) - (b[e] + np.linalg.norm(a) * bloat_factor) - EPS >= 0)
        conjunctions.append(sum([a[k]*x2[k] for k in range(len(x1))]) - (b[e] + np.linalg.norm(a) * bloat_factor) - EPS >= 0)
        disjunctions.append(conjunctions[0] and conjunctions[1])
    return sum(disjunctions) > 0

def plan(x_init, waypoints, x_bound, obstacles, bloat_factor=2., num_segs=None, MIPGap=None):
    curr = x_init
    path = [x_init,]
    for w in waypoints:
        print(curr,w)
        pert = 0.
        PWL = None
        while PWL is None:
            PWL = plan_gurobi(obstacles, curr, w, limits=x_bound, tmax=1e4, vmax=3., bloat_factor=bloat_factor+pert, num_segs=num_segs, MIPGap=MIPGap)
            pert = (np.random.rand()-0.5)*2.*1e-2
        path += PWL[1:]
        curr = w
    path = np.array(path)
    return path # N x 3

class Controller(object):
    """docstring for Controller"""
    def __init__(self, C3M_net, x_init=None, waypoints=None, x_bound=None, obstacles=None, bloat_factor=None, num_segs=None, MIPGap=None):
        super(Controller, self).__init__()
        self.C3M = C3M_net
        self.x_init = None
        self.waypoints = None
        self.x_bound = None
        self.obstacles = None
        self.bloat_factor = bloat_factor
        self.num_segs = num_segs
        self.MIPGap = MIPGap
        self.reset(x_init, waypoints, x_bound, obstacles, bloat_factor, num_segs, MIPGap)

    def reset(self, x_init, waypoints, x_bound, obstacles, bloat_factor, num_segs, MIPGap):
        if x_init is not None: self.x_init = x_init
        if waypoints is not None: self.waypoints = waypoints
        if x_bound is not None: self.x_bound = x_bound
        if obstacles is not None: self.obstacles = obstacles
        if bloat_factor is not None: self.bloat_factor = bloat_factor
        if num_segs is not None: self.num_segs = num_segs
        if MIPGap is not None: self.MIPGap = MIPGap
        if all(v is not None for v in [self.x_init, self.waypoints, self.x_bound, self.obstacles, self.bloat_factor, self.MIPGap]):
            data = [np.array(self.x_init).reshape(-1), np.array(self.waypoints).reshape(-1), np.array(self.x_bound).reshape(-1), np.array(self.bloat_factor).reshape(-1), np.array(self.MIPGap).reshape(-1)]
            for ob in self.obstacles:
                data.append(ob[0].reshape(-1))
                data.append(ob[1].reshape(-1))
            if self.num_segs is not None:
                data.append(np.array(self.num_segs).reshape(-1))
            data = np.concatenate(data)
            h = hashlib.md5(data)
            h = h.hexdigest()
            name = 'data/plannedpath/%s.pkl'%h
            if os.path.exists(name):
                with open(name, 'rb') as f:
                    self.path,self.waypoints,self.t,self.xref,self.uref = pickle.load(f)
            else:
                self.path = plan(self.x_init[:3], self.waypoints, self.x_bound, self.obstacles, self.bloat_factor, num_segs=self.num_segs, MIPGap=self.MIPGap)
                self.waypoints, self.t = interpolate(self.path)
                self.xref, self.uref = LQR.simulate(self.x_init, self.waypoints, self.t)
                with open(name, 'wb') as f:
                    pickle.dump([self.path,self.waypoints,self.t,self.xref,self.uref], f)

    def set_target(self, xcurr):
        dist = ((xcurr.reshape(1,-1) - self.xref)**2).sum(axis=1) # FIXME: use full state or positions only
        # eps = 1e-10
        eps = 0.
        threshold = dist.min() + eps
        self.idx = np.where(dist <= threshold)[0].max()
        self.local_t = self.t[self.idx]

    def __call__(self, xcurr, dt=None):
        if not hasattr(self, 'idx'):
            self.set_target(xcurr)
        xref = self.xref[self.idx, :]
        uref = self.uref[self.idx, :]
        xe = xcurr - xref
        u = self.C3M(xcurr, xe, uref)
        if dt == None:
            self.idx += 1
            if self.idx >= len(self.t):
                self.idx = len(self.t) - 1
            self.local_t = self.t[self.idx]
        else:
            self.local_t += dt
            if self.local_t > self.t[-1]:
                self.local_t = self.t[-1]
            dis = self.local_t - self.t
            dis[dis < 0] = np.inf
            self.idx = dis.argmin()
        return u, xref
