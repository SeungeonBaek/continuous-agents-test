import gym


class RLLoader():
    def __init__(self, env_config, agent_config):
        self.env_config = env_config
        self.agent_config = agent_config

    def env_loader(self):
        if self.env_config['env_name'] == 'LunarLanderContinuous-v2':
            env = gym.make(self.env_config['env_name'])
            obs_space = env.observation_space.shape
            act_space = env.action_space.shape
        elif self.env_config['env_name'] == 'pg-drive':
            env = gym.make('pg-drive')
            obs_space = env.observation_space.shape
            act_space = env.action_space.shape
        elif self.env_config['env_name'] == 'domestic':
            env = gym.make('nota-its-v0')
            obs_space = env.observation_space.shape
            act_space = env.action_space.shape

        return env, obs_space, act_space


    def agent_loader(self):
        if self.agent_config['agent_name'] == 'DDPG':
            if self.agent_config['extension']['name'] == 'TQC':
                from agents.DDPG_TQC import Agent
            elif self.agent_config['extension']['name'] == 'gSDE':
                from agents.DDPG_gSDE import Agent
            else:
                from agents.DDPG import Agent

        elif self.agent_config['agent_name'] == 'TD3':
            if self.agent_config['extension']['name'] == 'TQC':
                from agents.TD3_TQC import Agent
            elif self.agent_config['extension']['name'] == 'gSDE':
                from agents.TD3_gSDE import Agent
            else:
                from agents.TD3 import Agent

        elif self.agent_config['agent_name'] == 'PPO':
            if self.agent_config['extension']['name'] == 'Model_Ensemble':
                from agents.MEPPO import Agent
            elif self.agent_config['extension']['name'] == 'gSDE':
                from agents.PPO_gSDE import Agent
            else:
                from agents.PPO import Agent

        elif self.agent_config['agent_name'] == 'SAC':
            if self.agent_config['extension']['name'] == 'TQC':
                from agents.SAC_TQC import Agent
            elif self.agent_config['extension']['name'] == 'gSDE':
                from agents.SAC_gSDE import Agent
            else:
                from agents.SAC import Agent

        else:
            raise ValueError('Please try to set the correct Agent')

        return Agent