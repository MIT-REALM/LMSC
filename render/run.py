import pickle
import numpy as np
import sys
import time

from evaluate import *
from calc_metrics import get_distance_between_agents, get_distance_between_agents_and_obs

from ol_dynamics import f_batch_azlast

args = parse_args()
if args.scene == 'delivery':
    from scene.delivery import get_scene
elif args.scene == 'tunnel':
    from scene.tunnel import get_scene
else:
    raise ValueError('wrong scene name')

np.random.seed(args.seed)

num_agents = args.num_agents
obstacles, obstacles_old_representation, area_bound, waypoints = get_scene(num_agents, bloat_factor=args.bloat_factor)

# start and end
start_points = [ws[0] for ws in waypoints]
end_points = [ws[-1] for ws in waypoints]

from LQR import x_lb, x_ub
x_l = np.tile(x_lb.reshape(1,-1),(num_agents, 1))
x_u = np.tile(x_ub.reshape(1,-1),(num_agents, 1))

def simulate(area_bound, obstacles, waypoints, vis, rate, removal=False):
    s, u_ref, u, is_safe, loss_list, acc_list = build_evaluation_graph(args.num_agents)

    vars = tf.trainable_variables()
    vars_restore = []
    for v in vars:
        if 'action' in v.name or 'cbf' in v.name:
            vars_restore.append(v)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=vars_restore)
    saver.restore(sess, args.model_path)

    scene = macbf.Maze(args.num_agents)
    # set obs/goals/initials
    scene.bloat_factor = args.bloat_factor
    scene.MIPGap = args.MIPGap
    scene.num_segs = args.num_segs
    scene.area_bound = np.array(area_bound)
    scene.obstacles = obstacles
    scene.waypoints = waypoints
    scene.reset_reference_controllers()

    # scene.reset()
    start_time = time.time()
    s_np = np.concatenate(
        [[ws[0] for ws in scene.waypoints], np.zeros((args.num_agents, 5))], axis=1)
    s_traj = []
    trajectory = [s_np,]
    controls = []
    dist_agents = []
    dist_obs = []
    dists = []
    tracking_errors = []
    safety_info = np.zeros(args.num_agents, dtype=np.float32)
    is_safe_np = np.ones(shape=(args.num_agents, 1))
    prev_is_safe_np = is_safe_np

    cnt = 0
    TIME_HORIZON = args.time_horizon
    goal_tolerance = 3.
    active_agents = np.ones(num_agents, dtype=bool)

    noise = np.zeros([TIME_HORIZON, num_agents, 8])
    # noise_level = 0.1
    noise_level = args.noise_level
    num_segs = int(TIME_HORIZON / (3. / config.TIME_STEP_EVAL))
    segments = []
    for i in range(num_agents):
        segments.append(np.random.permutation(TIME_HORIZON)[:num_segs-1])
        segments[-1].sort()
        segments[-1] = [0,] + segments[-1].tolist() + [-1,]
    # print(segments)

    for i in range(num_segs):
        for j in range(num_agents):
            n = np.random.randn(1, 8) * noise_level
            noise[segments[j][i]:segments[j][i+1],j,:] = n

    stime = time.time()
    while True:
        # print(time.time()-stime)
        stime = time.time()
        print(cnt)
        cnt += 1
        if cnt >= TIME_HORIZON:
            print(np.sqrt(((s_np[:,:3] - np.array(end_points))**2).sum(axis=1)))
            print((np.sqrt(((s_np[active_agents,:3] - np.array(end_points)[active_agents,:])**2).sum(axis=1))))
            print(active_agents.sum())
            break

        if (np.sqrt(((s_np[active_agents,:3] - np.array(end_points)[active_agents,:])**2).sum(axis=1)) < goal_tolerance).sum() == active_agents.sum():
            print(np.sqrt(((s_np[:,:3] - np.array(end_points))**2).sum(axis=1)))
            print((np.sqrt(((s_np[active_agents,:3] - np.array(end_points)[active_agents,:])**2).sum(axis=1))))
            print(active_agents.sum())
            break

        # Get reference control input from the C3M controller
        u_ref_np, s_ref_np = scene.get_reference_control(s_np)

        if args.cbf != 'none':
            # Compute the final control input u_np with the CBF controller
            u_np, is_safe_np, acc_list_np = sess.run(
                [u, is_safe, acc_list], feed_dict={s:s_np, u_ref: u_ref_np})
        else:
            u_np = u_ref_np

        if args.cbf == 'full':
            idx = []
        elif args.cbf == 'threshold':
            # disable CBF for safe agents
            idx = np.where(is_safe_np)[0]
        elif args.cbf == 'none':
            # disable CBF for all agents
            idx = range(args.num_agents)
        else:
            raise ValueError('wrong args.cbf')
        for i in idx:
            u_np[i,:] = u_ref_np[i, :]

        dsdt = f_batch_azlast(s_np, u_np)
        dsdt += noise[cnt, :, :]
        s_np = s_np + dsdt * config.TIME_STEP_EVAL
        s_np[s_np<x_l] = x_l[s_np<x_l]
        s_np[s_np>x_u] = x_u[s_np>x_u]

        controls.append(u_np)
        tracking_error = np.sqrt(((s_ref_np-s_np)[:,:3]**2).sum(axis=1))
        tracking_error[~active_agents] = np.nan
        tracking_errors.append(tracking_error)

        # check safety
        dist_agents.append(get_distance_between_agents(np.expand_dims(s_np,0)).reshape(-1))
        dist_obs.append(get_distance_between_agents_and_obs(np.expand_dims(s_np,0), obstacles_old_representation).reshape(-1))
        dists.append(np.minimum(dist_agents[-1], dist_obs[-1]))

        if is_safe_np.sum()<s_np.shape[0]:
            print('iss:', is_safe_np.astype('int').reshape(-1).tolist())

        idx = np.where(is_safe_np.astype('int') - prev_is_safe_np.astype('int') > 0)[0]
        for i in idx:
        # for i in range(len(scene.controllers)):
            scene.controllers[i].set_target(s_np[i,:])
        prev_is_safe_np = is_safe_np
        trajectory.append(s_np)

    return trajectory, scene.controllers, controls, dists, dist_obs, dist_agents, tracking_errors

def main():
    traj, controllers, controls, dists, dist_obs, dist_agents, tracking_errors = simulate(area_bound, obstacles, waypoints, None, None, args.removal)
    traj = np.array(traj)
    path = [c.path for c in controllers]
    lqr = [c.xref for c in controllers]
    c3m = [traj[:,idx,:] for idx in range(num_agents)]
    import pickle
    pickle.dump({'obs':obstacles, 'start_points':start_points, 'end_points':end_points, 'x_lb':x_lb, 'x_ub':x_ub, 'traj':traj, 'lqr':lqr, 'path':path, 'dists': dists, 'dist_obs':dist_obs, 'dist_agents':dist_agents, 'tracking_errors':tracking_errors}, open(args.output, "wb" ))
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, traj.shape[1])]
    # from IPython import embed; embed()
    exit()

    from plan_and_track import plot_env
    plot_env(start_points, end_points, obstacles, path, lqr, c3m, colors)

    for i in range(num_agents):
        plot_env(start_points, end_points, obstacles, path[i:i+1], lqr[i:i+1], c3m[i:i+1], colors[i:i+1])

    X = np.expand_dims(traj[:,:,:3], -1) # T x N x 3 x 1
    error = (X - X.transpose([0,3,2,1])) # T x N x 3 x N
    error = np.sqrt((error**2).sum(axis=2)) # T x N x N
    error[:, np.eye(num_agents).astype('bool')] = np.inf
    error.min(axis=0)
    error[0,:].min()
    print(error.min())

    obs = np.array(obstacles).T # 6 x O
    X = np.expand_dims(traj[:,:,:3], -1) # T x N x 3 x 1
    X = np.concatenate([X>obs[:3,:], X<obs[3:,:]], axis=2) # T x N x 6 x O
    X = (X.sum(axis=2)==6) # T x N x O
    print(np.where(X>0)[0])
    print(X.sum(axis=0).sum(axis=1))
if __name__ == '__main__':
    main()
