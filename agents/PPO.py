import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LayerNormalization

from utils.replay_buffer_PPO import ExperienceMemory
from copy import deepcopy


class Actor(Model):
    def __init__(self, obs_space, action_space):
        super(Actor,self).__init__()
        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)

        self.l1 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l1_ln = LayerNormalization(axis=-1)
        self.l2 = Dense(128, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_ln = LayerNormalization(axis=-1)
        self.l3 = Dense(64, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3_ln = LayerNormalization(axis=-1)
        self.l4 = Dense(32, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l4_ln = LayerNormalization(axis=-1)
        self.mu = Dense(action_space, activation='tanh')
        self.std = Dense(action_space, activation='tanh')

    def call(self, state):
        l1 = self.l1_ln(self.l1(state))
        l2 = self.l2_ln(self.l2(l1))
        l3 = self.l3_ln(self.l3(l2))
        l4 = self.l4_ln(self.l4(l3))
        mu = self.mu(l4)
        std = self.std(l4)

        return mu, std


class Critic(Model):
    def __init__(self):
        super(Critic,self).__init__()
        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)
        
        self.l1 = Dense(256, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l1_ln = LayerNormalization(axis=-1)
        self.l2 = Dense(128, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_ln = LayerNormalization(axis=-1)
        self.l3 = Dense(64, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3_ln = LayerNormalization(axis=-1)
        self.l4 = Dense(32, activation = 'relu' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l4_ln = LayerNormalization(axis=-1)
        self.value = Dense(1, activation = None)

    def call(self, state):
        l1 = self.l1_ln(self.l1(state))
        l2 = self.l1_ln(self.l2(l1))
        l3 = self.l1_ln(self.l3(l2))
        l4 = self.l1_ln(self.l4(l3))
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

        self.replay_buffer = ExperienceMemory(2000)
        self.batch_size = self.agent_config['batch_size']
        self.epoch = self.agent_config['epoch_num']

        self.entropy_coeff = self.agent_config['entropy_coeff']
        self.entropy_reduction_rate = self.agent_config['entropy_reduction_rate']
        self.epsilon = self.agent_config['epsilon']

        self.reward_normalize = self.agent_config['reward_normalize']

        # network config
        self.actor_lr = self.agent_config['lr_actor']
        self.critic_lr = self.agent_config['lr_critic']

        self.actor = Actor(self.obs_space, self.act_space)
        self.actor_opt = Adam(self.actor_lr)
        self.actor.compile(optimizer=self.actor_opt)
        
        self.critic = Critic()
        self.critic_opt = Adam(self.critic_lr)
        self.critic.compile(optimizer=self.critic_opt)

        # extension config
        self.extension_config = self.agent_config['extension']
        self.extension_name = self.extension_config['name']
        self.std_bound = self.extension_config['std_bound']

    def action(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)

        mu, std_ = self.actor(obs)
        std = tf.stop_gradient(tf.clip_by_value(std_, clip_value_min=self.std_bound[0], clip_value_max=self.std_bound[1]))
        dist = tfp.distributions.Normal(loc=mu, scale=tf.math.abs(std))
        action = tf.squeeze(dist.sample())
        log_policy = tf.reduce_sum(dist.log_prob(action), 1, keepdims=False)

        action = action.numpy()
        action = np.clip(action, -1, 1)

        # print(f'in get action, state : {state.shape}')
        # print(f'mu, std : {mu.shape}, {std.shape}')
        # print(f'action : {action}')
        # print(f'log_policy : {log_policy}')

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
        update = True
        self.update_step += 1

        states, next_states, rewards, actions, old_log_policies, dones = self.replay_buffer.sample()

        states      = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards     = np.array(rewards, dtype=np.float32)
        actions     = np.array(actions, dtype=np.float32)
        old_log_polices   = np.array(old_log_policies, dtype=np.float32)
        dones       = np.array(dones, dtype=np.float32)

        values      = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))
        values      = np.array(values)

        next_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))
        next_values = np.array(next_values)

        if self.reward_normalize:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        advs, targets = self.get_gaes(rewards=rewards,dones=dones,values=values,next_values=next_values)

        # for logging
        entropy_mem     = 0
        ratio_mem       = 0
        actor_loss_mem  = 0
        adv_mem         = 0
        target_val_mem  = 0
        current_val_mem = 0
        critic_loss_mem = 0

        for _ in range(self.epoch):
            sample_range = np.arange(len(states))
            np.random.shuffle(sample_range)
            sample_idx = sample_range[:self.batch_size]
            
            batch_state = [states[i] for i in sample_idx]
            batch_action = [actions[i] for i in sample_idx]
            batch_target = [targets[i] for i in sample_idx]
            batch_old_log_policy = [old_log_polices[i] for i in sample_idx]
            batch_adv = [advs[i] for i in sample_idx]

            critic_variable = self.critic.trainable_variables
            with tf.GradientTape() as tape_critic:
                tape_critic.watch(critic_variable)
                current_value = self.critic(tf.convert_to_tensor(batch_state, dtype=tf.float32))
                # print(f'current_value : {current_value.shape}')
                current_value = tf.squeeze(current_value)
                # print(f'current_value : {current_value.shape}')

                target_value = tf.convert_to_tensor(batch_target, dtype=tf.float32)
                # print(f'target_value : {target_value.shape}')
                target_value = tf.squeeze(target_value)
                # print(f'target_value : {target_value.shape}')

                td_error = tf.subtract(current_value, target_value)
                # print(f'td_error : {td_error.shape}')

                critic_loss = tf.math.reduce_mean(tf.math.square(td_error))
                # print(f'critic_loss : {critic_loss.shape}')

            grads_critic, _ = tf.clip_by_global_norm(tape_critic.gradient(critic_loss, critic_variable), 0.5)

            self.critic_opt.apply_gradients(zip(grads_critic, critic_variable))


            actor_variable = self.actor.trainable_variables   
            with tf.GradientTape() as tape:
                tape.watch(actor_variable)
                train_mu, train_std = self.actor(tf.convert_to_tensor(batch_state, dtype=tf.float32))
                # print(f'train_mu : {train_mu.shape}, train_std : {train_std.shape}')

                train_dist = tfp.distributions.Normal(loc = train_mu, scale = tf.math.abs(train_std))

                entropy = tf.reduce_mean(train_dist.entropy())
                # print(f'entropy : {entropy.shape}')

                train_action = tf.convert_to_tensor(batch_action, dtype=tf.float32)
                # print(f'train_action : {train_action.shape}')

                train_log_policy = tf.reduce_sum(train_dist.log_prob(train_action), 1, keepdims=False)
                # print(f'train_log_policy : {train_log_policy.shape}')

                train_old_log_policy = tf.convert_to_tensor(batch_old_log_policy, dtype=tf.float32)
                # print(f'train_old_log_policy : {train_old_log_policy.shape}')
                train_old_log_policy = tf.squeeze(batch_old_log_policy)
                # print(f'train_old_log_policy : {train_old_log_policy.shape}')

                train_adv = tf.convert_to_tensor(batch_adv, dtype=tf.float32)
                # print(f'train_adv : {train_adv.shape}')
                train_adv = tf.squeeze(train_adv)
                # print(f'train_adv : {train_adv.shape}')

                ratio = tf.exp(train_log_policy - train_old_log_policy)
                # print(f'ratio : {ratio.shape}')

                surr1 = tf.multiply(train_adv, ratio)
                # print(f'surr1 : {surr1.shape}')
                surr2 = tf.multiply(train_adv, tf.clip_by_value(ratio, clip_value_min=1-self.epsilon, clip_value_max=1+self.epsilon))
                # print(f'surr2 : {surr2.shape}')

                minimum = tf.minimum(surr1, surr2)
                # print(f'minimum : {minimum.shape}')

                actor_loss = -tf.reduce_mean(minimum) - entropy * self.entropy_coeff
                # print(f'actor_loss : {actor_loss.shape}')
                
            grads, _ = tf.clip_by_global_norm(tape.gradient(actor_loss, actor_variable), 0.5)

            self.actor_opt.apply_gradients(zip(grads, actor_variable))

            advantage = tf.reduce_mean(train_adv).numpy()
            target_value = tf.reduce_mean(target_value).numpy()
            current_value = tf.reduce_mean(current_value).numpy()

            entropy_mem += entropy.numpy() / self.epoch
            ratio_mem += tf.reduce_mean(ratio).numpy() / self.epoch
            actor_loss_mem += actor_loss.numpy() / self.epoch
            adv_mem += advantage / self.epoch
            target_val_mem += target_value / self.epoch
            current_val_mem += current_value / self.epoch
            critic_loss_mem += critic_loss.numpy() / self.epoch
            
        self.entropy_coeff *= self.entropy_reduction_rate

        return True, entropy_mem, ratio_mem, actor_loss_mem, adv_mem, target_val_mem, current_val_mem, critic_loss_mem

    def save_xp(self, state, next_state, reward, action, log_policy, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add((state, next_state, reward, action, log_policy, done))

    def load_models(self, path):
        print('Load Model Path : ', path)
        self.actor.load_weights(path, "_actor")
        self.critic.load_weights(path, "_critic")

    def save_models(self, path, score):
        save_path = str(path) + "score_" + str(score) + "_model"
        print('Save Model Path : ', save_path)
        self.actor.save_weights(save_path, "_actor")
        self.critic.save_weights(save_path, "_critic")

