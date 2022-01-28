import pickle
import numpy as np
from calc_metrics import get_distance_between_agents_and_obs

with open('data/tunnel.pkl', 'rb') as f:
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
area_bound = [[0, 147], [-137, 130], [-100, 40]]

def get_scene(num_agents, bloat_factor=2., mutual_min_dist=1.2, clearance_waypoints=100.):
    waypoints = []
    B = np.array(area_bound)
    obstacles_old_representation = bboxs.reshape([bboxs.shape[0], -1])

    num_per_line = 2
    num_layers = int(np.ceil(num_agents / (num_per_line ** 2)))
    start = np.meshgrid(np.linspace(126, 146, num_layers), np.linspace(-135+5, -81-5, num_per_line), np.linspace(-18+5, 38-5, num_per_line), indexing='ij')
    start_points = list(np.stack([x.reshape(-1) for x in start], axis=1))
    start_points = start_points[:num_agents]

    end = np.meshgrid(np.linspace(2+3, 28-3, num_per_line), np.linspace(105, 125, num_layers), np.linspace(2+3, 18-3, num_per_line), indexing='ij')
    end_points = list(np.stack([x.reshape(-1) for x in end], axis=1))
    end_points = end_points[:num_agents]
    for s,e  in zip(start_points, end_points):
        rand = lambda: (np.random.rand(3)-0.5)*2
        path = [s, (np.array([122, -107, 10])+rand()*np.array([0,6,6])),
                    (np.array([22, -107, 10])+rand()*np.array([6,6,6])),
                    (np.array([22, -107, -90])+rand()*np.array([6,6,6])),
                    (np.array([22, -10, -90])+rand()*np.array([6,6,6])),
                    (np.array([22, -10, 10])+rand()*np.array([6,6,6])),
                    (np.array([22, 0.5, 10])+rand()*np.array([6,0,6])), e]
        waypoints.append(path)
    return obstacles, obstacles_old_representation, area_bound, waypoints
