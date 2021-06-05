import sys
sys.dont_write_bytecode = True

import os
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

import macbf
import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=12)
    parser.add_argument('--time_horizon', type=int, default=30000)
    parser.add_argument('--num_segs', type=int, default=None)
    parser.add_argument('--MIPGap', type=float, default=1e-4)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--vis', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ref', type=str, default=None)
    parser.add_argument('--noise_level', type=float, default=0.)
    parser.add_argument('--bloat_factor', type=float, default=1.)
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--cbf', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--removal', dest='removal', action='store_true')
    parser.set_defaults(removal=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args


def build_evaluation_graph(num_agents):
    s = tf.placeholder(tf.float32, [num_agents, 8])
    u_ref = tf.placeholder(tf.float32, [num_agents, 3])
    
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    h, mask, indices = macbf.network_cbf(
        x=x, r=config.DIST_MIN_THRES, indices=None)

    # We directly use the u from CCM, then refine u using CBF
    u = u_ref * 0.08

    safe_mask = macbf.compute_safe_mask(s, r=config.DIST_SAFE, indices=indices)
    is_safe = tf.equal(tf.reduce_mean(tf.cast(safe_mask, tf.float32)), 1)
    is_safe_each = tf.equal(tf.reduce_mean(
        tf.cast(safe_mask, tf.float32), axis=1), 1)

    u_res = tf.Variable(tf.zeros_like(u), name='u_res')
    loop_count = tf.Variable(0, name='loop_count')
    
    with tf.control_dependencies([
        u_res.assign(tf.zeros_like(u)), loop_count.assign(0)]):

        dsdt = macbf.quadrotor_dynamics_tf(s, u + u_res)
        s_next = s + dsdt * config.TIME_STEP_EVAL
        x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
        h_next, mask_next, _ = macbf.network_cbf(
            x=x_next, r=config.DIST_MIN_THRES, indices=indices)
        deriv = h_next - h + config.TIME_STEP_EVAL * config.ALPHA_CBF * h
        deriv = deriv * mask * mask_next
        error = tf.reduce_sum(tf.math.maximum(-deriv, 0), axis=1)
        error_gradient = tf.gradients(error, u_res)[0]
        
        error_gradient_squared = tf.reduce_sum(error_gradient**2, axis=1, keepdims=True)
        u_res = -error / (error_gradient_squared + 1e-12) * error_gradient
        u_res = tf.clip_by_value(u_res, -1, 1)
        u_opt = u + u_res

    loss_dang, loss_safe, acc_dang, acc_safe = macbf.loss_barrier(
        h=h, s=s, indices=indices)
    (loss_dang_deriv, loss_safe_deriv, loss_medium_deriv, acc_dang_deriv, 
    acc_safe_deriv, acc_medium_deriv) = macbf.loss_derivatives(
        s=s, u=u_opt, h=h, x=x, indices=indices)

    loss_action = macbf.loss_actions(s=s, u=u_opt, u_ref=u_ref, indices=indices)

    loss_list = [loss_dang, loss_safe, loss_dang_deriv, 
                 loss_safe_deriv, loss_medium_deriv, loss_action]
    acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv, acc_medium_deriv]

    return s, u_ref, u_opt, is_safe_each, loss_list, acc_list

    
def print_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_i = acc[:, i]
        acc_list.append(np.mean(acc_i[acc_i > 0]))
    print('Accuracy: {}'.format(acc_list))


def render_init(num_agents):
    fig = plt.figure(figsize=(10, 7))
    return fig


def show_obstacles(obs, ax, z=[0, 6], alpha=0.6, color='deepskyblue'):
    for x1, y1, x2, y2 in obs:
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


def clip_norm(x, thres):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    mask = (norm > thres).astype(np.float32)
    x = x * (1 - mask) + x * mask / (1e-6 + norm)
    return x


def clip_state(s, x_thres, v_thres=0.1, h_thres=6):
    x, v, r = s[:, :3], s[:, 3:6], s[:, 6:]
    x = np.concatenate([np.clip(x[:, :2], 0, x_thres),
                        np.clip(x[:, 2:], 0, h_thres)], axis=1)
    v = clip_norm(v, v_thres)
    s = np.concatenate([x, v, r], axis=1)
    return s


def main():
    args = parse_args()
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

    safety_ratios_epoch = []
    safety_ratios_epoch_baseline = []

    dist_errors = []
    dist_errors_baseline = []
    accuracy_lists = []

    if args.vis > 0:
        plt.ion()
        plt.close()
        fig = render_init(args.num_agents)

    scene = macbf.Maze(args.num_agents)
    if args.ref is not None:
        scene.read(args.ref)

    if not os.path.exists('trajectory'):
        os.mkdir('trajectory')
    traj_dict = {'ours': [], 'baseline': [], 'obstacles': [np.array(scene.OBSTACLES)]}

    safety_reward = []
    dist_reward = []
 
    for istep in range(config.EVALUATE_STEPS):
        if args.vis > 0:
            plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=80, azim=-45)
            ax.axis('off')
            show_obstacles(scene.OBSTACLES, ax)

        scene.reset()
        start_time = time.time()
        s_np = np.concatenate(
            [scene.start_points, np.zeros((args.num_agents, 5))], axis=1)
        s_traj = []
        safety_info = np.zeros(args.num_agents, dtype=np.float32)
        is_safe_np = np.ones(shape=(args.num_agents, 1))

        for i in range(config.INNER_LOOPS_EVAL):
            # Get reference control input from the C3M controller
            u_ref_np, s_ref_np = scene.get_reference_control(s_np)
            # Disable the C3M control if the agent is unsafe
            # u_ref_np = u_ref_np * is_safe_np + (1-is_safe_np) * u_ref_np * 0.1
            # Compute the final control input u_np with the CBF controller
            u_np, is_safe_np, acc_list_np = sess.run(
                [u, is_safe, acc_list], feed_dict={s:s_np, u_ref: u_ref_np})
            dsdt = macbf.quadrotor_dynamics_np(s_np, u_np)
            s_np = s_np + dsdt * config.TIME_STEP_EVAL
            safety_ratio = 1 - np.mean(
                macbf.dangerous_mask_np(s_np, config.DIST_MIN_CHECK), axis=1)
            individual_safety = safety_ratio == 1
            safety_info = safety_info + individual_safety - 1
            safety_ratio = np.mean(individual_safety)
            safety_ratios_epoch.append(safety_ratio)
            accuracy_lists.append(acc_list_np)

            # Visualize 1 out of 10 frames
            if args.vis == 1 and np.mod(i, 10) == 0:
                colors = []
                for j in range(individual_safety.shape[0]):
                    if individual_safety[j] == 1:
                        # Orange if the agent is safe
                        colors.append('darkorange')
                    else:
                        # Blue if the agent is unsafe (collision)
                        colors.append('darkblue')
                
                ax.set_xlim(0, 20)
                ax.set_ylim(0, 20)
                ax.set_zlim(0, 10)
                ax.scatter(s_np[:, 0], s_np[:, 1], s_np[0, 2], color=colors)
                for side in ax.spines.keys():
                    ax.spines[side].set_linewidth(2)
                    ax.spines[side].set_color('grey')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                fig.canvas.draw()

            s_traj.append(np.expand_dims(s_np[:, [0, 1, 2, 6, 7]], axis=0))
                
        safety_reward.append(np.mean(safety_info))
        dist_reward.append(np.mean((np.linalg.norm(
            s_np[:, :3] - s_ref_np[:, :3], axis=1) < 1.5).astype(np.float32) * 10))
        dist_errors.append(
            np.mean(np.linalg.norm(s_np[:, :3] - s_ref_np[:, :3], axis=1)))
        traj_dict['ours'].append(np.concatenate(s_traj, axis=0))
        end_time = time.time()

        s_np = np.concatenate(
            [scene.start_points, np.zeros((args.num_agents, 5))], axis=1)
        for k, c in enumerate(scene.controllers):
            c.set_target(s_np[k])

        s_traj = []
       
        for i in range(config.INNER_LOOPS_EVAL):
            u_ref_np, s_ref_np = scene.get_reference_control(s_np)
            u_np = u_ref_np
            dsdt = macbf.quadrotor_dynamics_np(s_np, u_np)
            s_np = s_np + dsdt * config.TIME_STEP_EVAL
            safety_ratio = 1 - np.mean(
                macbf.dangerous_mask_np(s_np, config.DIST_MIN_CHECK), axis=1)
            individual_safety = safety_ratio == 1
            safety_ratio = np.mean(individual_safety)
            safety_ratios_epoch_baseline.append(safety_ratio)

            if args.vis == 2 and np.mod(i, 10) == 0:
                colors = []
                for j in range(individual_safety.shape[0]):
                    if individual_safety[j] == 1:
                        colors.append('darkorange')
                    else:
                        colors.append('darkblue')

                ax.set_xlim(0, 20)
                ax.set_ylim(0, 20)
                ax.set_zlim(0, 10)
                ax.scatter(s_np[:, 0], s_np[:, 1], s_np[0, 2], color=colors)
                for side in ax.spines.keys():
                    ax.spines[side].set_linewidth(2)
                    ax.spines[side].set_color('grey')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                fig.canvas.draw()
                time.sleep((end_time - start_time) / config.INNER_LOOPS_EVAL)
            s_traj.append(np.expand_dims(s_np[:, [0, 1, 2, 6, 7]], axis=0))
        dist_errors_baseline.append(np.mean(np.linalg.norm(s_np[:, :3] - s_ref_np[:, :3], axis=1)))
        traj_dict['baseline'].append(np.concatenate(s_traj, axis=0))
        print('Evaluation Step: {} | {}, Time: {:.4f}'.format(
            istep + 1, config.EVALUATE_STEPS, end_time - start_time))

    print_accuracy(accuracy_lists)
    print('Distance Error (Learning | Baseline): {:.4f} | {:.4f}'.format(
          np.mean(dist_errors), np.mean(dist_errors_baseline)))
    print('Mean Safety Ratio (Learning | Baseline): {:.4f} | {:.4f}'.format(
          np.mean(safety_ratios_epoch), np.mean(safety_ratios_epoch_baseline)))

    safety_reward = np.mean(safety_reward)
    dist_reward = np.mean(dist_reward)
    print('Safety Reward: {:.4f}, Dist Reward: {:.4f}, Reward: {:.4f}'.format(
        safety_reward, dist_reward, 9 + 0.1 * (safety_reward + dist_reward)))

    pickle.dump(traj_dict, open('trajectory/traj_eval.pkl', 'wb'))
    scene.write_trajectory('trajectory/env_traj_eval.pkl', traj_dict['ours'])


if __name__ == '__main__':
    main()