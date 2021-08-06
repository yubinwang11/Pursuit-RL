import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import torch

from multiagent.environment import MultiAgentEnv
from utils.policies import BasePolicy
import envs.mpe_scenarios as scenarios
from algorithms.attention_sac import AttentionSAC

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="multi_speaker_listener", help="Name of environment")
    parser.add_argument("--model_name", default="multi_speaker_listener", 
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=1, type=int) # CHANGE THIS! default=12
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
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

    config = parser.parse_args()

    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='multi_speaker_listener.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    #model = torch.load('/home/admin888/MAAC-master/models/multi_speaker_listener/multi_speaker_listener/run35/model.pt')
    model = AttentionSAC.init_from_save('/home/admin888/MAAC-master/models/multi_speaker_listener/multi_speaker_listener/run35/model.pt')
    '''
    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)
    '''

    # execution loop
    obs_n = env.reset()
    
    while True:
        # query for action from each agent's policy
        act_n = []
        act_n = model.step(obs_n, explore=True)
        #for i, policy in enumerate(policies):
            #act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
