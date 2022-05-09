import os
from datetime import datetime

import numpy as np
import gym
import pandas as pd

from tensorboardX import SummaryWriter

def env_agent_setting(env_switch, agent_switch):
    if env_switch == 1:
        env_config = {'env_name': 'LunarLanderContinuous-v2', 'seed': 777, 'render': False, 'max_step': 500, 'max_episode': 2500}
    elif env_switch == 2: # Todo
        env_config = {'env_name': 'pg-drive', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    elif env_switch == 3: # Todo
        env_config = {'env_name': 'domestic-env', 'seed': 777, 'render': False, 'max_step': 1000, 'max_episode': 501}
    else:
        raise ValueError('Please try to correct env_switch')

    # DDPG
    if agent_switch == 1:
        from agent_config import DDPG_Vanilla_agent_config
        agent_config = DDPG_Vanilla_agent_config
    elif agent_switch == 2:
        from agent_config import DDPG_TQC_agent_config
        agent_config = DDPG_TQC_agent_config
    elif agent_switch == 3:
        from agent_config import DDPG_gSDE_agent_config
        agent_config = DDPG_gSDE_agent_config

    # TD3
    elif agent_switch == 4:
        from agent_config import TD3_Vanilla_agent_config
        agent_config = TD3_Vanilla_agent_config
    elif agent_switch == 5:
        from agent_config import TD3_TQC_agent_config
        agent_config = TD3_TQC_agent_config
    elif agent_switch == 6:
        from agent_config import TD3_gSDE_agent_config
        agent_config = TD3_gSDE_agent_config

    # PPO
    elif agent_switch == 7:
        from agent_config import PPO_Vanilla_agent_config
        agent_config = PPO_Vanilla_agent_config
    elif agent_switch == 8:
        from agent_config import ME_PPO_agent_config
        agent_config = ME_PPO_agent_config
    elif agent_switch == 9:
        from agent_config import PPO_gSDE_agent_config
        agent_config = PPO_gSDE_agent_config

    # SAC
    elif agent_switch == 10:
        from agent_config import SAC_Vanilla_agent_config
        agent_config = SAC_Vanilla_agent_config
    elif agent_switch == 11:
        from agent_config import SAC_TQC_agent_config
        agent_config = SAC_TQC_agent_config
    elif agent_switch == 12:
        from agent_config import SAC_gSDE_agent_config
        agent_config = SAC_gSDE_agent_config

    else:
        raise ValueError('Please try to correct agent_switch')
    
    return env_config, agent_config


def env_loader(env_config):
    if env_config['env_name'] == 'LunarLanderContinuous-v2':
        env = gym.make(env_config['env_name'])
        obs_space = env.observation_space.shape
        act_space = env.action_space.shape.n
    elif env_config['env_name'] == 'pg-drive':
        env = gym.make('pg-drive')
        obs_space = env.observation_space.shape
        act_space = env.action_space.shape.n
    elif env_config['env_name'] == 'domestic':
        env = gym.make('nota-its-v0')
        obs_space = env.observation_space.shape
        act_space = env.action_space.shape.n

    return env, obs_space, act_space


def agent_loader(agent_config):
    if agent_config['agent_name'] == 'DDPG':
        if agent_config['extension']['name'] == 'TQC':
            from agents.DDPG_TQC import Agent
        elif agent_config['extension']['name'] == 'gSDE':
            from agents.DDPG_gSDE import Agent
        else:
            from agents.DDPG import Agent

    elif agent_config['agent_name'] == 'TD3':
        if agent_config['extension']['name'] == 'TQC':
            from agents.TD3_TQC import Agent
        elif agent_config['extension']['name'] == 'gSDE':
            from agents.TD3_gSDE import Agent
        else:
            from agents.TD3 import Agent

    elif agent_config['agent_name'] == 'PPO':
        if agent_config['extension']['name'] == 'Model_Ensemble':
            from agents.MEPPO import Agent
        elif agent_config['extension']['name'] == 'gSDE':
            from agents.PPO_gSDE import Agent
        else:
            from agents.PPO import Agent

    elif agent_config['agent_name'] == 'SAC':
        if agent_config['extension']['name'] == 'TQC':
            from agents.SAC_TQC import Agent
        elif agent_config['extension']['name'] == 'gSDE':
            from agents.SAC_gSDE import Agent
        else:
            from agents.SAC import Agent

    else:
        raise ValueError('Please try to set the correct Agent')

    return Agent


def step_logging_tensorboard(agent_config, Agent, summary_writer, episode_step):
    if agent_config['agent_name'] == 'DDPG':
        if agent_config['extension']['name'] == 'TQC':
            updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
        elif agent_config['extension']['name'] == 'gSDE':
            updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()
        else:
            updated, actor_loss, critic_loss, trgt_q_mean, critic_value= Agent.update()

    elif agent_config['agent_name'] == 'TD3':
        if agent_config['extension']['name'] == 'TQC':
            updated, actor_loss, critic_loss, trgt_q_mean, critic_1_value, critic_2_value = Agent.update()
        elif agent_config['extension']['name'] == 'gSDE':
            updated, actor_loss, critic_loss, trgt_q_mean, critic_1_value, critic_2_value = Agent.update()
        else:
            updated, actor_loss, critic_loss, trgt_q_mean, critic_1_value, critic_2_value = Agent.update()

    elif agent_config['agent_name'] == 'PPO':
        if agent_config['extension']['name'] == 'MEPPO':
            updated, actor_loss, entropy, critic_loss, advantage, critic_value = Agent.update()
        elif agent_config['extension']['name'] == 'gSDE':
            updated, actor_loss, entropy, critic_loss, advantage, critic_value = Agent.update()
        else:
            updated, actor_loss, entropy, critic_loss, advantage, critic_value = Agent.update()

    elif agent_config['agent_name'] == 'SAC':
        if agent_config['extension']['name'] == 'TQC':
            updated, actor_loss, critic_loss, trgt_q_mean, critic_value, critic_q_value = Agent.update()
        elif agent_config['extension']['name'] == 'gSDE':
            updated, actor_loss, critic_loss, trgt_q_mean, critic_value, critic_q_value = Agent.update()
        else:
            updated, actor_loss, critic_loss, trgt_q_mean, critic_value, critic_q_value = Agent.update()

    if agent_config['agent_name'] == 'DDPG':
        if updated:
            summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
            summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Critic_value', critic_value, Agent.update_step)
    elif agent_config['agent_name'] == 'TD3':
        if updated:
            summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
            summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Critic_1_value', critic_1_value, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Critic_2_value', critic_2_value, Agent.update_step)
    elif agent_config['agent_name'] == 'PPO':
        if updated:
            summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
            summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Advantage', advantage, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Critic_value', critic_value, Agent.update_step)
            summary_writer.add_scalar('03_Actor/Entropy', entropy, Agent.update_step)
    elif agent_config['agent_name'] == 'SAC':
        if updated:
            summary_writer.add_scalar('01_Loss/Critic_1_loss', critic_loss, Agent.update_step)
            summary_writer.add_scalar('01_Loss/Actor_loss', actor_loss, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Critic_1_value', critic_value, Agent.update_step)


def step_logging_wandb(agent_config, Agent, summary_writer, episode_step):
    if agent_config['agent_name'] == 'DDPG':
        if agent_config['extension']['name'] == 'TQC':
            updated, critic_loss, trgt_q_mean, critic_value= Agent.update(episode_step)
        elif agent_config['extension']['name'] == 'gSDE':
            updated, critic_loss, trgt_q_mean, critic_value= Agent.update(episode_step)
        else:
            updated, critic_loss, trgt_q_mean, critic_value= Agent.update(episode_step)

    elif agent_config['agent_name'] == 'TD3':
        updated, critic_1_loss, critic_2_loss, trgt_q_mean, critic_1_value, critic_2_value = Agent.update()

    elif agent_config['agent_name'] == 'PPO':
        updated, critic_1_loss, critic_2_loss, trgt_q_mean, critic_1_value, critic_2_value = Agent.update()

    elif agent_config['agent_name'] == 'SAC':
        updated, critic_1_loss, critic_2_loss, trgt_q_mean, critic_1_value, critic_2_value = Agent.update()

    if agent_config['agent_name'] == 'DDPG':
        if updated:
            summary_writer.add_scalar('01_Loss/Critic_loss', critic_loss, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Critic_value', critic_value, Agent.update_step)
    elif agent_config['agent_name'] == 'TD3':
        if updated:
            summary_writer.add_scalar('01_Loss/Critic_1_loss', critic_loss, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Critic_1_value', critic_1_value, Agent.update_step)
    elif agent_config['agent_name'] == 'PPO':
        if updated:
            summary_writer.add_scalar('01_Loss/Critic_1_loss', critic_1_loss, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Critic_1_value', critic_1_value, Agent.update_step)
    elif agent_config['agent_name'] == 'SAC':
        if updated:
            summary_writer.add_scalar('01_Loss/Critic_1_loss', critic_1_loss, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Target_Q_mean', trgt_q_mean, Agent.update_step)
            summary_writer.add_scalar('02_Critic/Critic_1_value', critic_1_value, Agent.update_step)


def episode_logging(summary_writer, episode_score, episode_step, episode_num):
    summary_writer.add_scalar('00_Episode/Score', episode_score, episode_num)
    summary_writer.add_scalar('00_Episode/Average_reward', episode_score/episode_step, episode_num)
    summary_writer.add_scalar('00_Episode/Steps', episode_step, episode_num)


def main(env_config, agent_config, rl_confing, summary_writer, data_save_path):
    # Env
    env, env_obs_space, env_act_space = env_loader(env_config)
    print(f"env_name : {env_config['env_name']}, obs_space : {env_obs_space}, act_space : {env_act_space}")

    if len(env_obs_space) > 1:
        for space in env_obs_space:
            obs_space *= space
    else:
        obs_space = env_obs_space[0]

    act_space = env_act_space[0]

    # Agent
    print(agent_config)
    RLAgent = agent_loader(agent_config)
    Agent = RLAgent(agent_config, obs_space, act_space)
    print('agent_name: {}'.format(agent_config['agent_name']))

    # csv logging
    if rl_confing['csv_logging']:
        episode_data = dict()
        episode_data['episode_score'] = np.zeros(env_config['max_episode'], dtype=np.float32)
        episode_data['mean_reward']   = np.zeros(env_config['max_episode'], dtype=np.float32)
        episode_data['episode_step']  = np.zeros(env_config['max_episode'], dtype=np.float32)

    for episode_num in range(1, env_config['max_episode']):
        episode_score = 0
        episode_step = 0
        done = False

        prev_obs = None
        prev_action = None

        obs = env.reset()
        obs = np.array(obs)
        obs = obs.reshape(-1)

        action = None

        while not done:
            if env_config['render']:
                env.render()
            episode_step += 1

            action = Agent.action(obs)
            
            obs, reward, done, _ = env.step(action)
            obs = np.array(obs)
            obs = obs.reshape(-1)

            action = np.array(action)

            episode_score += reward

            # Save_xp
            if episode_step > 2:
                Agent.save_xp(prev_obs, obs, reward, prev_action, done)

            prev_obs = obs
            prev_action = action

            if episode_step >= env_config['max_step']:
                done = True
                continue
            
            step_logging_tensorboard(agent_config, Agent, summary_writer, episode_step)

        env.close()

        episode_logging(summary_writer, episode_score, episode_step, episode_num)

        if rl_confing['csv_logging']:
            episode_data['episode_score'][episode_num-1] = episode_score
            episode_data['mean_reward'][episode_num-1]   = episode_score/episode_step
            episode_data['episode_step'][episode_num-1]  = episode_step

            if episode_num % 10 == 0:
                episode_data_df = pd.DataFrame(episode_data)
                episode_data_df.to_csv(data_save_path+'episode_data.csv', mode='w',encoding='UTF-8' ,compression=None)

        print('epi_num : {episode}, epi_step : {step}, score : {score}, mean_reward : {mean_reward}'.format(episode= episode_num, step= episode_step, score = episode_score, mean_reward=episode_score/episode_step))
        
    env.close()

if __name__ == '__main__':
    # Env =>   1: LunarLander-v2, 2: pg-drive 3: dometstic env
    # Agent => 1: DDPG, 2: DDPG_TQC, 3: DDPG_gSDE    #  4: TD3,  5: TD3_TQC,  6: TD3_gSDE    #  7: PPO,  8: MEPPO,    9: PPO_gSDE    # 10: SAC, 11: SAC_TQC, 12: SAC_gSDE
    env_switch = 1
    agent_switch = 4

    env_config, agent_config = env_agent_setting(env_switch, agent_switch)
    
    rl_config = {'csv_logging': False}

    parent_path = str(os.path.abspath(''))
    time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    result_path = parent_path + '/results/{env}/{agent}_result/'.format(env=env_config['env_name'], agent=agent_config['agent_name']) + time_string
    data_save_path = parent_path + '\\results\\{env}\\{agent}_result\\'.format(env=env_config['env_name'], agent=agent_config['agent_name']) + time_string + '\\'
    summary_writer = SummaryWriter(result_path+'/tensorboard/')

    main(env_config, agent_config, rl_config, summary_writer, data_save_path)