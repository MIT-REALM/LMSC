import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Loader(object):
    
    def __init__(self, data_path):
        self.data = pickle.load(open(data_path, 'rb'))
        self.load_obstacles()
        self.load_start_end_points()

    def load_obstacles(self):
        """ Load the obstacles.
        Obstacles: [[x1, y1, x2, y2, h],
                    [x1, y1, x2, y2, h], 
                    ...
                    [x1, y1, x2, y2, h]]
        (x1, y1) and (x2, y2) are diagonal points of the cubic obstacles.
        """
        self.obstacles = self.data['obstacles']

    def load_start_end_points(self):
        """ Load the starting and end points of the trajectoies.
        Starting points: (NEpisodes, NAgents, 3)
        End points: (NEpisodes, NAgents, 3)
        """
        self.start_points = self.data['start_points']
        self.end_points = self.data['end_points']

    def show_obstacles(self, obs, ax, alpha=0.6, color='grey'):
        """ Show the obstacles in the 3D space
        Args:
            obs (N, 5): N obstacles [x1, y1, x2, y2, h].
            ax: The matplotlib drawing handle.
            alpha (float): Transparancy of the obstacles.
            color (str): Color of the obstacles.
        """
        for x1, y1, x2, y2, h in obs:
            z = [0, h] 
            xs, ys = np.meshgrid([x1, x2], [y1, y2])
            zs = np.ones_like(xs)
            ax.plot_surface(xs, ys, zs * z[0], alpha=alpha, color=color)
            ax.plot_surface(xs, ys, zs * z[1], alpha=alpha, color=color)

            xs, zs = np.meshgrid([x1, x2], z)
            ys = np.ones_like(xs)
            ax.plot_surface(xs, ys * y1, zs, alpha=alpha, color=color)
            ax.plot_surface(xs, ys * y2, zs, alpha=alpha, color=color)

            ys, zs = np.meshgrid([y1, y2], z)
            xs = np.ones_like(ys)
            ax.plot_surface(xs * x1, ys, zs, alpha=alpha, color=color)
            ax.plot_surface(xs * x2, ys, zs, alpha=alpha, color=color)


if __name__ == '__main__':

    loader = Loader('./save/scene.pkl')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=80, azim=-45)
    ax.axis('off')

    loader.show_obstacles(loader.obstacles, ax)

    # Randomly choose an episode to show the start and end points
    index = np.random.randint(loader.start_points.shape[0])
    start = loader.start_points[index]
    end = loader.end_points[index]

    ax.scatter(start[:, 0], start[:, 1], start[:, 2], color='darkorange', label='Start')
    ax.scatter(end[:, 0], end[:, 1], end[:, 2], color='darkred', label='End')

    plt.legend()
    plt.tight_layout()
    plt.show()