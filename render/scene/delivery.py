import pickle
import numpy as np
from calc_metrics import get_distance_between_agents_and_obs

# np.random.seed(0)
# env
with open('data/delivery.pkl', 'rb') as f:
    bboxs = pickle.load(f)

obstacles = []
for bbox in bboxs:
    A = np.array([[-1, 0,  0],
                  [ 1, 0,  0],
                  [0, -1,  0],
                  [0,  1,  0],
                  [0,  0, -1],
                  [0,  0,  1]])
    b = bbox.T.reshape(-1) * np.array([-1,1,-1,1,-1,1])
    obstacles.append([A, b])
bboxs = np.array(bboxs)
area_bound = np.array([bboxs[:,0,:].min(axis=0), bboxs[:,1,:].max(axis=0)]).T.tolist()
area_bound[2][1] = 11.

def get_scene(num_agents, bloat_factor=2., mutual_min_dist=1.2, clearance_waypoints=100.):
    waypoints = []
    B = np.array(area_bound)
    obstacles_old_representation = bboxs.reshape([bboxs.shape[0], -1])

    bloat_factor = bloat_factor * 1.1
    num_waypoints = 3
    num_samples_trial = 100
    for _ in range(num_agents):
        waypoints.append([])
        for i in range(num_waypoints):
            while True:
                p = B[:,0:1] + bloat_factor + (B[:,1:2] - B[:,0:1] - 2 * bloat_factor) * np.random.rand(B.shape[0], num_samples_trial)
                p = p.T # num_samples_trial x n
                dist = get_distance_between_agents_and_obs(p.reshape(num_samples_trial, 1, B.shape[0]), obstacles_old_representation).squeeze()
                conditions = [dist > bloat_factor * np.sqrt(B.shape[0]),]

                if len(waypoints[-1]) > 0:
                    conditions.append(np.sqrt(((p - waypoints[-1][-1].reshape(1, -1))**2).sum(axis=1)) < clearance_waypoints)
                if i == 0 or i == num_waypoints-1:
                    if len(waypoints) > 1:
                        others = np.array([w[i] for w in waypoints[:-1]]) # m x n
                        dist = p.reshape(num_samples_trial, 1, B.shape[0]) - others.reshape(1, others.shape[0], others.shape[1]) # num_samples_trial x m x n
                        dist = np.sqrt((dist**2).sum(axis=2)).min(axis=1)
                        conditions.append(dist > mutual_min_dist)
                conditions = np.logical_and.reduce(np.array(conditions), axis=0)
                idx = np.where(conditions)[0]
                if len(idx) > 0:
                    break
            waypoints[-1].append(p[idx[0], :])
    return obstacles, obstacles_old_representation, area_bound, waypoints
