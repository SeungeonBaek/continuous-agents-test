import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense

from utils.replay_buffer import ExperienceMemory
from copy import deepcopy


class Actor(Model):
    def __init__(self, obs_space, action_space):
        super(Actor,self).__init__()
        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)

        self.l1 = Dense(64, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(128, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3 = Dense(64, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l4 = Dense(32, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.mu = Dense(action_space, activation='tanh')
        self.std = Dense(action_space, activation='tanh')

    def call(self, state):
        l1 = self.l1(state)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        mu = self.mu(l4)
        std = self.std(l4)

        return mu, std


class Critic(Model):
    def __init__(self):
        super(Critic,self).__init__()
        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)
        
        self.l1 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(128, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l4 = Dense(32, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.value = Dense(1, activation = None)

    def call(self, state_action):
        l1 = self.l1(state_action)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        value = self.value(l4)

        return value


class Agent:
    """
    Argument:
        agent_config: agent configuration which is realted with RL algorithm => DDPG
            agent_config:
                {
                    name, gamma, update_freq, batch_size, epoch_num, eps_clip, eps_reduction_rate,
                    lr_actor, lr_critic, buffer_size, use_GAE, lambda,  reward_normalize
                    extension = {
                        'gaussian_std, 'noise_clip, 'noise_reduce_rate'
                    }
                }
        obs_shape_n: shpae of observation
        act_shape_n: shape of action

    Methods:
        action: return the action which is mapped with obs in policy
        target_action: return the target action which is mapped with obs in target_policy
        update_target: update target critic/actor network at user-specified frequency
        update: update main critic/actor network
        save_xp: save transition(s, a, r, s', d) in experience memory
        load_models: load weights
        save_models: save weights
    
    """
    def __init__(self, agent_config, obs_space, act_space):
        self.agent_config = agent_config
        self.name = self.agent_config['agent_name']

        self.obs_space = obs_space
        self.act_space = act_space
        print(f'obs_space: {self.obs_space}, act_space: {self.act_space}')

        self.gamma = self.agent_config['gamma']
        self.lamda = self.agent_config['lambda']
        self.gae_normalize = self.agent_config['use_GAE']

        self.update_step = 0

        self.replay_buffer = ExperienceMemory(self.agent_config['512'])
        self.batch_size = self.agent_config['batch_size']
        self.epoch = self.agent_config['epoch_num']

        self.entropy_coeff = self.agent_config['entropy_coeff']
        self.entropy_reduction_rate = self.agent_config['entropy_reduction_rate']
        self.epsilon = self.agent_config['epsilon']

        # extension config
        self.extension_config = self.agent_config['extension']
        self.std_bound = self.extension_config['std_bound']

        self.actor_lr = self.agent_config['lr_actor']
        self.critic_lr = self.agent_config['lr_critic']

        self.actor = Actor(self.act_space)
        self.actor_opt = Adam(self.actor_lr)
        self.actor.compile(optimizer=self.actor_opt)
        
        self.critic = Critic()
        self.critic_opt = Adam(self.critic_lr)
        self.critic.compile(optimizer=self.critic_opt)

    def action(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)

        mu, std_ = self.actor(obs)
        std = tf.stop_gradient(tf.clip_by_value(std_, clip_value_min=self.std_bound[0], clip_value_max=self.std_bound[1]))
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        action = tf.squeeze(dist.sample())
        log_policy = tf.reduce_sum(dist.log_prob(action), 1, keepdims=False)

        action = action.numpy()
        action = np.clip(action, -1, 1)

        # print('in get action, state : ', state)
        # print('mu, std :', mu, std)
        # print('action : ', action)
        # print('log_policy : ', log_policy)

        return action, log_policy.numpy()
    
    def get_gaes(self, rewards, dones, values, next_values):
        deltas = [r + self.gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = deepcopy(deltas)

        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * self.gamma * self.lamda * gaes[t + 1]
        
        if gaes.std() >= 0:
            gaes_norm = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        else:
            gaes_norm = (gaes - gaes.mean()) / (gaes.std() - 1e-8)

        if self.gae_normalize:
            gaes = gaes_norm

        target = gaes + values

        return gaes, target

    def update(self):
        # print('entity_episodes : ', entity_episodes)
        state = np.array([], np.float64)
        next_state = np.array([], np.float64)
        reward = np.array([], np.float64)
        done = np.array([], np.float64)
        action = np.array([], np.int32)
        old_log_policy = np.array([], np.float64)
        adv = np.array([], np.float64)
        target = np.array([], np.float64)
        
        entity_episodes = None

        for _, entity_samples in entity_episodes.items():
            # print('entity_samples : ', entity_samples)
            _, _, next_value = self.ppo(tf.convert_to_tensor(entity_samples['next_state'], dtype=tf.float32))
            next_value = np.array(next_value)

            if self.reward_normalization:
                entity_samples['reward'] = (entity_samples['reward'] - entity_samples['reward'].mean()) / (entity_samples['reward'].std() + 1e-5)

            entity_adv, entity_target = self.get_gaes(
                rewards=np.squeeze(entity_samples['reward']),
                dones=np.squeeze(entity_samples['done']),
                values=np.squeeze(entity_samples['old_value']),
                next_values=np.squeeze(next_value))
            
            if state.size == 0:
                state = deepcopy(entity_samples['state'])
                next_state = deepcopy(entity_samples['next_state'])
                reward = deepcopy(entity_samples['reward'])
                done = deepcopy(entity_samples['done'])
                action = deepcopy(np.squeeze(entity_samples['action']))
                old_log_policy = deepcopy(np.squeeze(entity_samples['old_log_policy']))
                adv = deepcopy(entity_adv)
                target = deepcopy(entity_target)
            
            else:
                state = np.concatenate((state, entity_samples['state']),axis = 0)
                next_state = np.concatenate((next_state, entity_samples['next_state']),axis = 0)
                reward = np.concatenate((reward, entity_samples['reward']),axis = 0)
                done = np.concatenate((done, entity_samples['done']),axis = 0)
                action = np.concatenate((action, np.squeeze(entity_samples['action'])),axis = 0)
                old_log_policy = np.concatenate((old_log_policy, np.squeeze(entity_samples['old_log_policy'])),axis = 0)
                adv = np.concatenate((adv, entity_adv), axis = 0)
                target = np.concatenate((target, entity_target), axis = 0)

        # print('state : ', np.shape(state))
        # print('next_state : ', np.shape(next_state))
        # print('reward : ', np.shape(reward))
        # print('done : ', np.shape(done))
        # print('old_log_policy : ', np.shape(old_log_policy))
        # print('adv : ', np.shape(adv))
        # print('target : ', np.shape(target))
        # print('action : ', np.shape(action))

        total_loss = 0

        entropy_mem = 0
        ratio_mem = 0
        pi_loss_mem = 0
        adv_mem = 0
        value_target_mem = 0
        value_cur_mem = 0
        value_loss_mem = 0
        total_loss_mem = 0

        for _ in range(self.epoch):
            sample_range = np.arange(len(state))
            np.random.shuffle(sample_range)
            sample_idx = sample_range[:self.batch_size]
            
            batch_state = [state[i] for i in sample_idx]
            # batch_done = [done[i] for i in sample_idx]
            batch_action = [action[i] for i in sample_idx]
            batch_target = [target[i] for i in sample_idx]
            batch_old_log_policy = [old_log_policy[i] for i in sample_idx]
            batch_adv = [adv[i] for i in sample_idx]

            ppo_variable = self.ppo.trainable_variables
            
            with tf.GradientTape() as tape:
                tape.watch(ppo_variable)
                train_mu, train_std, train_current_value = self.ppo(tf.convert_to_tensor(batch_state, dtype=tf.float32))
                train_current_value = tf.squeeze(train_current_value)
                # 추가
                # train_std_ = tf.stop_gradient(tf.clip_by_value(train_std, clip_value_min=self.std_bound[0], clip_value_max=1))
                train_dist = tfp.distributions.Normal(loc = train_mu, scale = train_std)
                # print('train_mu : ', np.shape(train_mu.numpy()))
                # print('train_std : ', np.shape(train_std.numpy()))
                # print('train cur value : ', np.shape(train_current_value.numpy()))
                # print('train dist : ', train_dist.batch_shape_tensor())

                train_adv = tf.convert_to_tensor(batch_adv, dtype=tf.float32)
                # print('train adv : ', np.shape(train_adv.numpy()))

                train_target = tf.convert_to_tensor(batch_target, dtype=tf.float32)
                # print('train target : ', np.shape(train_target.numpy()))

                train_action = tf.convert_to_tensor(batch_action, dtype=tf.float32)
                # print('train action : ', np.shape(train_action.numpy()))

                # entropy = tf.reduce_mean(tf.multiply(-tf.squeeze(train_logits),tf.math.log(tf.squeeze(train_logits) + 1e-8)))
                entropy = tf.reduce_mean(train_dist.entropy())
                train_log_policy = tf.reduce_sum(train_dist.log_prob(train_action), 1, keepdims=False)
                train_old_log_policy = tf.convert_to_tensor(batch_old_log_policy, dtype=tf.float32)

                # print('train entropy : ', np.shape(entropy.numpy()))
                # print('train log policy : ', np.shape(train_log_policy.numpy()))
                # print('train old log policy : ', np.shape(train_old_log_policy.numpy()))

                ratio = tf.exp(train_log_policy - train_old_log_policy)
                surr1 = tf.multiply(train_adv, ratio)
                surr2 = tf.multiply(train_adv, tf.clip_by_value(ratio, clip_value_min=1-self.ppo_eps, clip_value_max=1+self.ppo_eps))
                minimum = tf.minimum(surr1, surr2)
                pi_loss = -tf.reduce_mean(minimum) - entropy * self.ppo_ent
                
                # print('train ratio : ', np.shape(ratio.numpy()))
                # print('train surr1 : ', np.shape(surr1.numpy()))
                # print('train surr1 : ', np.shape(surr1.numpy()))
                # print('train minimum : ', np.shape(minimum.numpy()))
                # print('train pi_loss : ', np.shape(pi_loss.numpy()))

                value_loss = tf.reduce_mean(tf.square(train_target - train_current_value))
                total_loss = pi_loss + self.ppo_val * value_loss
                
                # print('train value_loss : ', np.shape(value_loss.numpy()))
                # print('in ppo train total_loss : ', np.shape(total_loss.numpy()))

            if self.gradient_clipping:
                grads, _ = tf.clip_by_global_norm(tape.gradient(total_loss, ppo_variable), 0.5)
            else:
                grads = tape.gradient(total_loss, ppo_variable)

            self.opt.apply_gradients(zip(grads, ppo_variable))
            self.gradient_steps += 1

            advantage = tf.reduce_mean(train_adv).numpy()
            value_target = tf.reduce_mean(train_target).numpy()
            value_cur = tf.reduce_mean(train_current_value).numpy()

            # print('train advantage : ', np.shape(advantage))
            # print('train value_target : ', np.shape(value_target))
            # print('train value_cur : ', np.shape(value_cur))  

            entropy_mem += entropy.numpy() / self.epoch
            ratio_mem += tf.reduce_mean(ratio).numpy() / self.epoch
            pi_loss_mem += pi_loss.numpy() / self.epoch
            adv_mem += advantage / self.epoch
            value_target_mem += value_target / self.epoch
            value_cur_mem += value_cur / self.epoch
            value_loss_mem += value_loss.numpy() / self.epoch
            total_loss_mem += total_loss.numpy() / self.epoch
            
        self.entropy_coeff *= self.entropy_reduction_rate

        return True, entropy_mem, ratio_mem, pi_loss_mem, adv_mem, value_target_mem, value_cur_mem, value_loss_mem

    def save_xp(self, state, next_state, reward, action, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add((state, next_state, reward, action, done))

    def load_models(self, path):
        print('Load Model Path : ', path)
        self.actor.load_weights(path, "_actor")
        self.critic.load_weights(path, "_critic")

    def save_models(self, path, score):
        save_path = str(path) + "score_" + str(score) + "_model"
        print('Save Model Path : ', save_path)
        self.actor.save_weights(save_path, "_actor")
        self.critic.save_weights(save_path, "_critic")

