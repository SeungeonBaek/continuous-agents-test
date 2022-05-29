import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense

from utils.replay_buffer import ExperienceMemory
from utils.prioritized_memory_numpy import PrioritizedMemory

from agents.Noise_model import NoiseModel

class GaussianActor(Model):
    """
        Gaussian policy
    """
    def __init__(self, gaussian_actor_config, obs_space, action_space):
        super(GaussianActor,self).__init__()
        self.gaussian_actor_config = gaussian_actor_config

        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)

        self.l1 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

        self.noise = NoiseModel()
        self.noise.build((None, self.gaussian_actor_config['latent_space']))

        self.mu = Dense(action_space, activation='tanh')
        self.std = Dense(action_space, activation='tanh')

        self.bijector = tfp.bijectors.Tanh()

    def reset_noise(self):
        self.noise.sample_weights()

    def call(self, state):
        l1 = self.l1(state)
        l2 = self.l2(l1)
        mu = self.mu(l2)
        std = self.mu(l2)

        return mu, std


class ImplicitActor(Model):
    """
        Implicit policy
    """
    def __init__(self,
                 implicit_actor_config,
                 obs_space,
                 action_space):
        super(ImplicitActor,self).__init__()
        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)

        self.l1 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.mu = Dense(action_space, activation='tanh')
        self.std = Dense(action_space, activation='tanh')

    def call(self, state):
        l1 = self.l1(state)
        l2 = self.l2(l1)
        mu = self.mu(l2)
        std = self.mu(l2)

        return mu, std


class DistCritic(Model):
    """
        Implicit Distributional Critic
    """
    def __init__(self):
        super(DistCritic,self).__init__()
        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)

        self.l1 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.value = Dense(1, activation=None)

    def call(self, state, action, noise):
        l1 = self.l1(tf.concat([state, action, noise], axis=1))
        l2 = self.l2(l1)
        value = self.value(l2)

        return value

    def sample(self, state, action, num_saples=1)


class Agent:
    """
    Argument:
        agent_config: agent configuration which is realted with RL algorithm => DDPG
            agent_config:
                {
                    name, gamma, tau, update_freq, batch_size, warm_up, lr_actor, lr_critic,
                    buffer_size, use_PER, use_ERE, reward_normalize,
                    alpha, qualtile_dim, target_entropy
                    extension = {
                        use_implicit_actor, use_automatic_entropy_tuning
                        implicit_config: {
                            actor_noise_num, actor_noise_dim, log_sig_min, log_sig_max,
                            log_prob_min, log_prob_max
                        }
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
        self.tau = self.agent_config['tau']

        self.update_step = 0
        self.update_freq = self.agent_config['update_freq']
        
        if self.agent_config['use_PER']:
            self.replay_buffer = PrioritizedMemory(self.agent_config['buffer_size'])
        else:
            self.replay_buffer = ExperienceMemory(self.agent_config['buffer_size'])
        self.batch_size = self.agent_config['batch_size']
        self.warm_up = self.agent_config['warm_up']

        # network config
        self.actor_lr_main = self.agent_config['lr_actor']
        self.critic_lr_main = self.agent_config['lr_critic']

        # extension config
        self.extension_config = self.agent_config['extension']

        if self.extension_config['use_implicit_actor']:
            self.actor_main = ImplicitActor(self.extension_config['implicit_config'], self.obs_space, self.act_space)
            self.actor_opt_main = Adam(self.actor_lr_main)
            self.actor_main.compile(optimizer=self.actor_opt_main)
        else:
            self.actor_main = GaussianActor(self.obs_space, self.act_space)
            self.actor_opt_main = Adam(self.actor_lr_main)
            self.actor_main.compile(optimizer=self.actor_opt_main)

        self.critic_main, self.critic_target = DistCritic(), DistCritic()
        self.critic_target.set_weights(self.critic_main.get_weights())
        self.critic_opt_main = Adam(self.critic_lr_main)
        self.critic_main.compile(optimizer=self.critic_opt_main)

    def action(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        # print(f'in action, obs: {obs.shape}')
        mu = self.actor_main(obs)
        # print(f'in action, mu: {mu.shape}')

        if self.update_step > self.warm_up:
            std = tf.convert_to_tensor([self.std]*4, dtype=tf.float32)
            dist = tfp.distributions.Normal(loc=mu, scale=std)
            action = tf.squeeze(dist.sample())
            action = action.numpy()
            action = np.clip(action, mu.numpy()[0]-self.noise_clip, mu.numpy()[0]+self.noise_clip)

            self.std = self.std * self.reduce_rate
            # print(f'in action, action: {action}')
        else:
            action = mu.numpy()[0]
            # print(f'in action, action: {action}')

        action = np.clip(action, -1, 1)
        # print(f'in action, clipped action: {action}')
        
        return action

    def target_action(self, obs):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        # print(f'in trgt action, obs: {obs}')
        mu = self.actor_target(obs)
        # print(f'in trgt action, mu: {mu}')

        if self.update_step > self.warm_up:
            std = tf.convert_to_tensor([self.std]*4, dtype=tf.float32)
            dist = tfp.distributions.Normal(loc=mu, scale=std)
            action = tf.squeeze(dist.sample())

        action = mu.numpy()
        # print(f'in trgt action, action: {action}')
        action = np.clip(action, -1, 1)
        # print(f'in trgt action, clipped_action: {action}')

        return action

    def update_target(self):
        actor_weights = []
        actor_targets = self.actor_target.get_weights()
        
        for idx, weight in enumerate(self.actor_main.get_weights()):
            actor_weights.append(weight * self.tau + actor_targets[idx] * (1 - self.tau))
        self.actor_target.set_weights(actor_weights)
        
        critic_weithgs = []
        critic_targets = self.critic_target.get_weights()
        
        for idx, weight in enumerate(self.critic_main.get_weights()):
            critic_weithgs.append(weight * self.tau + critic_targets[idx] * (1 - self.tau))
        self.critic_target.set_weights(critic_weithgs)

    def update(self):
        if self.replay_buffer._len() < self.batch_size:
            return False, 0.0, 0.0, 0.0, 0.0
        if not self.update_step % self.update_freq == 0:  # only update every update_freq
            self.update_step += 1
            return False, 0.0, 0.0, 0.0, 0.0

        updated = True
        self.update_step += 1
        
        if self.agent_config['use_PER']:
            states, next_states, rewards, actions, dones, idxs, is_weight = self.replay_buffer.sample(self.batch_size)

            if self.agent_config['reward_normalize']:
                rewards = np.asarray(rewards)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            states = tf.convert_to_tensor(states, dtype = tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
            actions = tf.convert_to_tensor(actions, dtype = tf.float32)
            dones = tf.convert_to_tensor(dones, dtype = tf.bool)
            is_weight = tf.convert_to_tensor(is_weight, dtype=tf.float32)
            # print('states : {}'.format(states.numpy().shape), states)
            # print('next_states : {}'.format(next_states.numpy().shape), next_states)
            # print('rewards : {}'.format(rewards.numpy().shape), rewards)
            # print('actions : {}'.format(actions.numpy().shape), actions)
            # print('dones : {}'.format(dones.numpy().shape), dones)
            # print('is_weight : {}'.format(is_weight.numpy().shape), is_weight)

        else:
            states, next_states, rewards, actions, dones = self.replay_buffer.sample(self.batch_size)

            if self.agent_config['reward_normalize']:
                rewards = np.asarray(rewards)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            states = tf.convert_to_tensor(states, dtype = tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
            actions = tf.convert_to_tensor(actions, dtype = tf.float32)
            dones = tf.convert_to_tensor(dones, dtype = tf.bool)
        
        critic_variable = self.critic_main.trainable_variables
        with tf.GradientTape() as tape_critic:
            tape_critic.watch(critic_variable)
            target_action = self.target_action(next_states)
            # print(f'target_action : {target_action.shape}')

            target_q_next = tf.squeeze(self.critic_target(tf.concat([next_states,target_action], 1)), 1)
            # print(f'target_q_next : {target_q_next.shape}')

            target_q = rewards + self.gamma * target_q_next * (1.0 - tf.cast(dones, dtype=tf.float32))
            # print(f'target_q : {target_q.shape}')

            current_q = tf.squeeze(self.critic_main(tf.concat([states,actions], 1)), 1)
            # print(f'current_q : {current_q.shape}')
        
            td_error = tf.subtract(current_q, target_q)
            # print(f'td_error : {td_error.shape}')

            critic_losses = tf.cond(tf.convert_to_tensor(self.agent_config['use_PER'], dtype=tf.bool), \
                    lambda: tf.multiply(is_weight, tf.math.square(td_error)), \
                    lambda: tf.math.square(td_error))
            # print(f'critic_losses : {critic_losses.shape}')

            critic_loss = tf.math.reduce_mean(critic_losses)
            # print(f'critic_loss : {critic_loss.shape}')

        grads_critic, _ = tf.clip_by_global_norm(tape_critic.gradient(critic_loss, critic_variable), 0.5)

        self.critic_opt_main.apply_gradients(zip(grads_critic, critic_variable))

        actor_variable = self.actor_main.trainable_variables       
        with tf.GradientTape() as tape_actor:
            tape_actor.watch(actor_variable)

            new_policy_actions = self.actor_main(states)
            actor_loss = -self.critic_main(tf.concat([states, new_policy_actions],1))
            actor_loss = tf.math.reduce_mean(actor_loss)
            
        grads_actor, _ = tf.clip_by_global_norm(tape_actor.gradient(actor_loss, actor_variable), 0.5)
        self.actor_opt_main.apply_gradients(zip(grads_actor, actor_variable))

        target_q_val = tf.math.reduce_mean(target_q).numpy()
        current_q_val = tf.math.reduce_mean(current_q).numpy()
        criitic_loss_val = critic_loss.numpy()
        actor_loss_val = actor_loss.numpy()
        
        self.update_target()

        td_error_numpy = np.abs(td_error.numpy())
        if self.agent_config['use_PER']:
            for i in range(self.batch_size):
                self.replay_buffer.update(idxs[i], td_error_numpy[i])

        return updated, actor_loss_val, criitic_loss_val, target_q_val, current_q_val

    def save_xp(self, state, next_state, reward, action, done):
        # Store transition in the replay buffer.
        if self.agent_config['use_PER']:
            state_tf = tf.convert_to_tensor([state], dtype = tf.float32)
            action_tf = tf.convert_to_tensor([action], dtype = tf.float32)
            next_state_tf = tf.convert_to_tensor([next_state], dtype = tf.float32)
            target_action_tf = self.actor_target(next_state_tf)
            # print(f'state_tf: {state_tf}')
            # print(f'action_tf: {action_tf}')
            # print(f'next_state_tf: {next_state_tf}')
            # print(f'target_action_tf: {target_action_tf}')

            target_q_next = tf.squeeze(self.critic_target(tf.concat([next_state_tf,target_action_tf], 1)), 1)
            # print(f'target_q_next: {target_q_next}')
            
            current_q = tf.squeeze(self.critic_main(tf.concat([state_tf,action_tf], 1)), 1)
            # print(f'current_q_1: {current_q_1}')
            
            target_q = reward + self.gamma * target_q_next * (1.0 - tf.cast(done, dtype=tf.float32))
            # print(f'target_q: {target_q}')
            
            td_error = tf.subtract(target_q ,current_q)
            # print(f'td_error: {td_error}')

            td_error_numpy = np.abs(td_error)
            # print(f'td_error_numpy: {td_error_numpy}')

            self.replay_buffer.add(td_error_numpy[0], (state, next_state, reward, action, done))
        else:
            self.replay_buffer.add((state, next_state, reward, action, done))

    def load_models(self, path):
        print('Load Model Path : ', path)
        self.actor_main.load_weights(path, "_actor_main")
        self.actor_target.load_weights(path, "_actor_target")
        self.critic_main.load_weights(path, "_critic_main")
        self.critic_target.load_weights(path, "_critic_target")

    def save_models(self, path, score):
        save_path = str(path) + "score_" + str(score) + "_model"
        print('Save Model Path : ', save_path)
        self.actor_main.save_weights(save_path, "_actor_main")
        self.actor_target.save_weights(save_path, "_actor_target")
        self.critic_main.save_weights(save_path, "_critic_main")
        self.critic_target.save_weights(save_path, "_critic_target")