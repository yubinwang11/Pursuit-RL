import numpy as np
import seaborn as sns
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075/2 if agent.adversary else 0.05/2 #0.075 0.05
            agent.accel = 3/8 if agent.adversary else 4/8 #3 4
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0/10 if agent.adversary else 1.3/10 #1.0 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.02*5 # 0.02*5
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        ######self.reset_cached_rewards()
        return world
    '''
    def reset_cached_rewards(self):
        self.pair_rewards = None
    def post_step(self, world):
        self.reset_cached_rewards()
    '''

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            
            # CHANGE THIS！
            #agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_pos = np.random.uniform(-0.5, +0.5, world.dim_p)

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
        ##self.reset_cached_rewards()

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            #CHANGE THIS!
            '''
            if x < 0.9: # CHANGE THIS! 0.9
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
            '''
            if x < 0.7: 
                return 0
            if x < 1.0:
                return (x - 0.7) * 10
            return min(np.exp(2 * x - 2), 10)

        
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)


'''            
class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 5
        num_listeners = 4
        num_speakers = 4
        num_landmarks = 6
        world.landmark_colors = np.array(
            sns.color_palette(n_colors=num_landmarks))
        world.listeners = []
        for li in range(num_listeners):
            agent = Agent()
            agent.i = li
            agent.name = 'agent %i' % agent.i
            agent.listener = True
            agent.collide = False
            agent.size = 0.075
            agent.silent = True
            agent.accel = 1.5
            agent.initial_mass = 1.0
            agent.max_speed = 1.0
            world.listeners.append(agent)
        world.speakers = []
        for si in range(num_speakers):
            agent = Agent()
            agent.i = si + num_listeners
            agent.name = 'agent %i' % agent.i
            agent.listener = False
            agent.collide = False
            agent.size = 0.075
            agent.movable = False
            agent.accel = 1.5
            agent.initial_mass = 1.0
            agent.max_speed = 1.0
            world.speakers.append(agent)
        world.agents = world.listeners + world.speakers
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.i = i + num_listeners + num_speakers
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
            landmark.color = world.landmark_colors[i]
        # make initial conditions
        self.reset_world(world)
        self.reset_cached_rewards()
        return world

    def reset_cached_rewards(self):
        self.pair_rewards = None

    def post_step(self, world):
        self.reset_cached_rewards()

    def reset_world(self, world):
        listen_inds = list(range(len(world.listeners)))
        np.random.shuffle(listen_inds)  # randomize which listener each episode
        for i, speaker in enumerate(world.speakers):
            li = listen_inds[i]
            speaker.listen_ind = li
            speaker.goal_a = world.listeners[li]
            speaker.goal_b = np.random.choice(world.landmarks)
            speaker.color = np.array([0.25,0.25,0.25])
            world.listeners[li].color = speaker.goal_b.color + np.array([0.25, 0.25, 0.25])
            world.listeners[li].speak_ind = i

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.reset_cached_rewards()

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return reward(agent, world)

    def calc_rewards(self, world):
        rews = []
        for speaker in world.speakers:
            dist = np.sum(np.square(speaker.goal_a.state.p_pos -
                                    speaker.goal_b.state.p_pos))
            rew = -dist
            if dist < (speaker.goal_a.size + speaker.goal_b.size) * 1.5:
                rew += 10.
            rews.append(rew)
        return rews

    def reward(self, agent, world):
        if self.pair_rewards is None:
            self.pair_rewards = self.calc_rewards(world)
        share_rews = False
        if share_rews:
            return sum(self.pair_rewards)
        if agent.listener:
            return self.pair_rewards[agent.speak_ind]
        else:
            return self.pair_rewards[agent.goal_a.speak_ind]

    def observation(self, agent, world):
        if agent.listener:
            obs = []
            # give listener index of their speaker
            obs += [agent.speak_ind == np.arange(len(world.speakers))]
            # give listener communication from its speaker
            obs += [world.speakers[agent.speak_ind].state.c]
            # give listener its own position/velocity,
            obs += [agent.state.p_pos, agent.state.p_vel]

            # obs += [world.speakers[agent.speak_ind].state.c]
            # # # give listener index of their speaker
            # # obs += [agent.speak_ind == np.arange(len(world.speakers))]
            # # # give listener all communications
            # # obs += [speaker.state.c for speaker in world.speakers]
            # # give listener its own velocity
            # obs += [agent.state.p_vel]
            # # give listener locations of all agents
            # # obs += [a.state.p_pos for a in world.agents]
            # # give listener locations of all landmarks
            # obs += [l.state.p_pos for l in world.landmarks]
            return np.concatenate(obs)
        else:  # speaker
            obs = []
            # give speaker index of their listener
            obs += [agent.listen_ind == np.arange(len(world.listeners))]
            # speaker gets position of listener and goal
            obs += [agent.goal_a.state.p_pos, agent.goal_b.state.p_pos]

            # # give speaker index of their listener
            # # obs += [agent.listen_ind == np.arange(len(world.listeners))]
            # # # give speaker all communications
            # # obs += [speaker.state.c for speaker in world.speakers]
            # # give speaker their goal color
            # obs += [agent.goal_b.color]
            # # give speaker their listener's position
            # obs += [agent.goal_a.state.p_pos]
            #
            # obs += [speaker.state.c for speaker in world.speakers]
            return np.concatenate(obs)    
'''