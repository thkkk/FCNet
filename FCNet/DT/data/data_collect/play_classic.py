# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_robot_personal import LEGGED_ROBOT_PERSONAL_ROOT_DIR
import os

import isaacgym
from legged_robot_personal.envs import task_registry, get_args
from legged_gym.utils import export_policy_as_jit, Logger

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 256)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = args.add_noise
    if hasattr(env_cfg, 'debug'):
        env_cfg.debug.debug_viz = args.debug_viz

    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = args.push_robots

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    writer = None
    if hasattr(task_registry, 'resume_path') and args.tensorboard:
        resume_path = '/'.join(task_registry.resume_path.split('/')[:-1])+f"/play_{datetime.now().strftime('%b%d_%H-%M-%S')}"
        writer = SummaryWriter(log_dir=resume_path, flush_secs=10)
    
    # export policy as a jit module (used to run it from C++)
    if args.export_policy:
        path = os.path.join(LEGGED_ROBOT_PERSONAL_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 300#100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([5., 0., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    obs = env.get_observations()

    for i in range(100*int(env.max_episode_length)):
        actions = policy(obs)
        rets = env.step(actions.detach())
        obs = rets[0]

        ews, dones, infos = rets[-3:]

        if args.record_frames:
            filename = os.path.join(LEGGED_ROBOT_PERSONAL_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', "%05d.bmp"%img_idx)
            if not os.path.exists(os.path.dirname(filename)):os.makedirs(os.path.dirname(filename))
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1 
        if args.move_camera:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)



        if writer is not None:
            contact_force = torch.norm(env.contact_forces, dim = -1)
            writer.add_scalars('actions', {f'{j}':actions[robot_index, j] for j in range(12)}, i)
            if hasattr(env_cfg.env, 'num_kd_actions') and env_cfg.env.num_kd_actions > 0: 
                writer.add_scalars('kd_actions', {f'{j}':actions[robot_index, 12:][j] for j in range(12)}, i)
            writer.add_scalars('dof_pos', {f'{j}':env.dof_pos[robot_index, j] for j in range(12)}, i)
            writer.add_scalars('dof_vel', {f'{j}':env.dof_vel[robot_index, j] for j in range(12)}, i)
            writer.add_scalars('dof_torque', {f'{j}':env.torques[robot_index, j] for j in range(12)}, i)
            writer.add_scalars('contact_force', {f'{j}':contact_force[robot_index, j] for j in range(contact_force.shape[1])}, i)
            writer.add_scalars('feet_contact_force_z', {f'{j}':env.contact_forces[robot_index, env.feet_indices[j], 2] for j in range(4)}, i)
            writer.add_scalars('base_vel', {f'{j}':env.base_lin_vel[robot_index, j] for j in range(3)}, i)
            writer.add_scalars('foot_vel_z', {f'{j}':env.rigid_body_velocity[robot_index, env.feet_indices[j], 2] for j in range(4)}, i)
            #writer.add_scalars('mechanical_power', {f'{j}':env.mechanical_energy[robot_index, j] for j in range(12)}, i)
            #writer.add_scalars('thermal_power', {f'{j}':env.thermal_energy[robot_index, j] for j in range(12)}, i)
            #writer.add_scalar('done', dones[robot_index], i)


        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_pos1': env.dof_pos[robot_index, joint_index+1].item(),
                    'dof_pos2': env.dof_pos[robot_index, joint_index+2].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_acc': (env.dof_vel[robot_index, joint_index] - env.last_dof_vel[robot_index, joint_index]).item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    args = get_args()
    play(args)
