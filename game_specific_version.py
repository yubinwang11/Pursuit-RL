import argparse
import torch
import time
import imageio
from imageio import imsave
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC
# CHANGE THIS!
#from envs.mpe_scenarios.multi_speaker_listener import Scenario
#from multiagent.core import World, Agent, Landmark
#from utils.environment import MultiAgentEnv


''''''
def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            #env.seed(seed + rank * 1000)
            #np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    '''
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            # CHANGE This! run_num = max(exst_run_nums) + 1 
            run_num = max(exst_run_nums)
            # CHANGE THIS！
            run_num -= 0
    '''
    run_num = 183
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run

    # add gifs saving path
    gif_path = model_dir.parent / 'gifs_eval'
    if config.save_gifs:
        gif_path.mkdir(exist_ok=True)

    fps = 10
    ifi = 1 / fps  # inter-frame interval

    #torch.manual_seed(run_num)
    #np.random.seed(run_num)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    #env = make_env(config.env_id, discrete_action=True)
    
    model = AttentionSAC.init_from_save(run_dir / 'model.pt')

    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        obs = env.reset()
        model.prep_rollouts(device='cpu')

        frames = []
              
        if config.save_gifs:
            frames.append(env.render('rgb_array')[0])

        for et_i in range(config.episode_length):

            # Save gifs
            calc_start = time.time()

            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)

            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            
            env.render('rgb_array')

            # CHANGE THIS！
            '''
            image = env.render('rgb_array')[0]
            imsave('et_i', image)
            '''

            if config.save_gifs:
                #image = env.render('rgb_array')[0]
                #frames.append(image)
                #image = env.render('rgb_array')[0]
                frames.append(env.render('rgb_array'))

            calc_end = time.time()
            elapsed = calc_end - calc_start

            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
          
            t += config.n_rollout_threads
  
    if config.save_gifs:
        gif_num = 0
        while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
            gif_num += 1

        imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))), frames, duration=ifi)
    


    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="multi_speaker_listener", help="Name of environment")
    parser.add_argument("--model_name", default="multi_speaker_listener", 
                        help="Name of directory to load " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=1, type=int) # CHANGE THIS! default=12
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=1, type=int) # CHANGE THIS! 50000
    parser.add_argument("--episode_length", default=1000, type=int) # CHANGE THIS! 25 25*1000000000
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true', default=True)
    
    # Add the argument of saving gifs
    parser.add_argument('--save_gifs', default=True, type=bool)

    config = parser.parse_args()

    run(config)
