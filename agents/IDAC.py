from cProfile import label
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import huber

from utils.replay_buffer import ExperienceMemory
from utils.prioritized_memory_numpy import PrioritizedMemory

tf.executing_eagerly()

class GaussianActor(Model):
    """
        Gaussian policy
    """
    def __init__(self, gaussian_actor_config, obs_space, action_space):
        super(GaussianActor,self).__init__()
        self.gaussian_actor_config = gaussian_actor_config

        self.obs_space = obs_space
        self.action_space = action_space

        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)

        self.l1 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

        if self.gaussian_actor_config['use_reparam_trick']:
            self.noise_dim  = self.gaussian_actor_config['noise_dim']
        else:
            pass

        self.log_sig_min = self.gaussian_actor_config['log_sig_min']
        self.log_sig_max = self.gaussian_actor_config['log_sig_max']
        self.log_prob_min = self.gaussian_actor_config['log_prob_min']
        self.log_prob_max = self.gaussian_actor_config['log_prob_max']

        self.mu = Dense(action_space, activation=None)
        self.std = Dense(action_space, activation=None)

        self.bijector = tfp.bijectors.Tanh()

    def call(self, state):
        if self.gaussian_actor_config['use_reparam_trick']:
            noise = tf.random.normal(shape=(state.shape[0] ,self.noise_dim),mean=0, stddev=1)
            print(f'in gaussian actor, in call, noise: {noise.shape}')

            l1 = self.l1(tf.concat([state, noise], axis=1))
            print(f'in gaussian actor, in call, l1: {l1.shape}')
            l2 = self.l2(l1)
            print(f'in gaussian actor, in call, l2: {l2.shape}')
            mu = self.mu(l2)
            print(f'in gaussian actor, in call, mu: {mu.shape}')
            std = self.std(l2)
            print(f'in gaussian actor, in call, std: {std.shape}')
            std = tf.squeeze(tf.exp(tf.clip_by_value(std[..., tf.newaxis], self.log_sig_min, self.log_sig_max)), axis=2)
            print(f'in gaussian actor, in call, std: {std.shape}')
            dist = tfp.distributions.TransformedDistribution(
                tfp.distributions.Normal(loc=mu, scale=std),
                bijector = self.bijector
                )

            action = tf.squeeze(tf.clip_by_value(dist.sample()[..., tf.newaxis], -1+1e-6, 1-1e-6))
            print(f'in gaussian actor, in call, action: {action.shape}')
            log_prob = tf.squeeze(tf.clip_by_value(dist.log_prob(action)[..., tf.newaxis], self.log_prob_min, self.log_prob_max), axis=2)
            print(f'in gaussian actor, in call, log_prob: {log_prob.shape}')
            log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
            print(f'in gaussian actor, in call, log_prob: {log_prob.shape}')

        else:
            # print(f'in gaussian actor, in call, state: {state.shape}')

            l1 = self.l1(state)
            # print(f'in gaussian actor, in call, l1: {l1.shape}')
            l2 = self.l2(l1)
            # print(f'in gaussian actor, in call, l2: {l2.shape}')
            mu = self.mu(l2)
            # print(f'in gaussian actor, in call, mu: {mu.shape}')
            std = self.std(l2)
            # print(f'in gaussian actor, in call, std: {std.shape}')
            std = tf.squeeze(tf.exp(tf.clip_by_value(std[..., tf.newaxis], self.log_sig_min, self.log_sig_max)), axis=2)
            # print(f'in gaussian actor, in call, std: {std.shape}')
            tf.debugging.check_numerics(std, message=f'std is not numeric, {std}')
            dist = tfp.distributions.TransformedDistribution(
                tfp.distributions.Normal(loc=mu, scale=std),
                bijector = self.bijector
                )

            action = tf.squeeze(tf.clip_by_value(dist.sample()[..., tf.newaxis], -1+1e-6, 1-1e-6))
            # print(f'in gaussian actor, in call, action: {action.shape}')
            log_prob = tf.squeeze(tf.clip_by_value(dist.log_prob(action)[..., tf.newaxis], self.log_prob_min, self.log_prob_max), axis=2)
            # print(f'in gaussian actor, in call, log_prob: {log_prob.shape}')
            tf.debugging.check_numerics(log_prob, message=f'log_prob is not numeric, \n action: {action}, \n log_prob: {log_prob}')
            log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
            # print(f'in gaussian actor, in call, log_prob: {log_prob.shape}')
            tf.debugging.check_numerics(log_prob, message=f'log_prob is not numeric, {log_prob}')

        return action, log_prob

    def sample(self, state):
        if self.gaussian_actor_config['use_reparam_trick']:
            noise = tf.random.normal(shape=(state.shape[0] ,self.noise_dim),mean=0, stddev=1)
            print(f'in gaussian actor, in sample, in sample, noise: {noise.shape}')

            l1 = self.l1(tf.concat([state, noise], axis=1))
            print(f'in gaussian actor, in sample, l1: {l1.shape}')
            l2 = self.l2(l1)
            print(f'in gaussian actor, in sample, l2: {l2.shape}')
            mu = self.mu(l2)
            print(f'in gaussian actor, in sample, mu: {mu.shape}')
            std = self.std(l2)
            print(f'in gaussian actor, in sample, std: {std.shape}')
            std = tf.squeeze(tf.exp(tf.clip_by_value(std[..., tf.newaxis], self.log_sig_min, self.log_sig_max)), axis=2)
            print(f'in gaussian actor, in sample, std: {std.shape}')
            dist = tfp.distributions.TransformedDistribution(
                tfp.distributions.Normal(loc=mu, scale=std),
                bijector = self.bijector
                )

            action = tf.squeeze(dist.sample())
            print(f'in gaussian actor, in sample, action: {action.shape}')

        else:
            l1 = self.l1(state)
            # print(f'in gaussian actor, in sample, l1: {l1.shape}')
            l2 = self.l2(l1)
            # print(f'in gaussian actor, in sample, l2: {l2.shape}')
            mu = self.mu(l2)
            # print(f'in gaussian actor, in sample, mu: {mu.shape}')
            std = self.std(l2)
            # print(f'in gaussian actor, in sample, std: {std.shape}')
            # print(f'in gaussian actor, in sample, std: {std}')
            std = tf.squeeze(tf.exp(tf.clip_by_value(std[..., tf.newaxis], self.log_sig_min, self.log_sig_max)), axis=2)
            # print(f'in gaussian actor, in sample, std: {std.shape}')
            # print(f'in gaussian actor, in sample, std: {std}')
            dist = tfp.distributions.TransformedDistribution(
                tfp.distributions.Normal(loc=mu, scale=std),
                bijector = self.bijector
                )

            action = tf.squeeze(dist.sample())
            # print(f'in gaussian actor, in sample, action: {action.shape}')

        return action


class ImplicitActor(Model):
    """
        Implicit policy
    """
    def __init__(self,
                 implicit_actor_config,
                 obs_space,
                 action_space):
        super(ImplicitActor,self).__init__()
        self.implicit_actor_config = implicit_actor_config

        self.noise_dim = self.implicit_actor_config['noise_dim']
        self.action_num = self.implicit_actor_config['action_num']

        self.log_sig_min = self.implicit_actor_config['log_sig_min']
        self.log_sig_max = self.implicit_actor_config['log_sig_max']

        self.log_prob_min = self.implicit_actor_config['log_prob_min']
        self.log_prob_max = self.implicit_actor_config['log_prob_max']

        self.obs_space = obs_space
        self.action_space = action_space

        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)

        self.l1 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

        self.mu = Dense(action_space, activation=None)
        self.std = Dense(action_space, activation=None)

        self.bijector = tfp.bijectors.Tanh()

    def call(self, state):
        # action 0
        noise = tf.random.normal(shape=(state.shape[0], self.noise_dim), mean=0, stddev=1)
        # print(f'in implicit actor, in call, noise: {noise.shape}')

        l1 = self.l1(tf.concat([state, noise], axis=1))
        # print(f'in implicit actor, in call, l1: {l1.shape}')
        l2 = self.l2(l1)
        # print(f'in implicit actor, in call, l2: {l2.shape}')
        mu = self.mu(l2)
        # print(f'in implicit actor, in call, mu: {mu.shape}')
        std = self.mu(l2)
        # print(f'in implicit actor, in call, std: {std.shape}')
        std = tf.squeeze(tf.exp(tf.clip_by_value(std[..., tf.newaxis], self.log_sig_min, self.log_sig_max)), axis=2)
        # print(f'in implicit actor, in call, std: {std.shape}')

        dist = tfp.distributions.TransformedDistribution(
            tfp.distributions.Normal(loc=mu, scale=std),
            bijector = self.bijector
            )
        action = tf.squeeze(tf.clip_by_value(dist.sample()[..., tf.newaxis], -1+1e-6, 1-1e-6))
        # print(f'in implicit actor, in call, action: {action.shape}')

        log_prob_main = tf.squeeze(tf.clip_by_value(dist.log_prob(action)[..., tf.newaxis], self.log_prob_min, self.log_prob_max), axis=2)
        # print(f'in implicit actor, in call, log_prob_main: {log_prob_main.shape}')
        tf.debugging.check_numerics(log_prob_main, message=f'log_prob_main is not numeric, {log_prob_main}')
        prob_main = tf.reduce_sum(tf.exp(log_prob_main), axis=1, keepdims=True)
        # print(f'in implicit actor, in call, prob_main: {prob_main.shape}')
        # tf.debugging.check_numerics(prob_main, message=f'prob_main is not numeric, {prob_main}')

        # actions
        state_repeat = tf.repeat(state, self.action_num, axis=0)
        # print(f'in implicit actor, in call, state_repeat: {state_repeat.shape}')
        action_repeat = tf.repeat(action, self.action_num, axis=0)
        # print(f'in implicit actor, in call, action_repeat: {action_repeat.shape}')

        noise_actions = tf.random.normal(shape=(state.shape[0] * self.action_num, self.noise_dim), mean=0, stddev=1)
        # print(f'in implicit actor, in call, noise_actions: {noise_actions.shape}')

        l1_actions = self.l1(tf.concat([state_repeat, noise_actions], axis=1))
        # print(f'in implicit actor, in call, l1_actions: {l1_actions.shape}')
        l2_actions = self.l2(l1_actions)
        # print(f'in implicit actor, in call, l2_actions: {l2_actions.shape}')
        mu_actions = self.mu(l2_actions)
        # print(f'in implicit actor, in call, mu_actions: {mu_actions.shape}')
        std_actions = self.mu(l2_actions)
        # print(f'in implicit actor, in call, std_actions: {std_actions.shape}')
        std_actions = tf.squeeze(tf.exp(tf.clip_by_value(std_actions[..., tf.newaxis], self.log_sig_min, self.log_sig_max)), axis=2)
        # print(f'in implicit actor, in call, std_actions: {std_actions.shape}')

        dist_actions = tfp.distributions.TransformedDistribution(
            tfp.distributions.Normal(loc=mu_actions, scale=std_actions),
            bijector = self.bijector
            )

        log_prob_actions = tf.squeeze(tf.clip_by_value(dist_actions.log_prob(action_repeat)[..., tf.newaxis], self.log_prob_min, self.log_prob_max), axis=2)
        # print(f'in implicit actor, in call, log_prob_actions: {log_prob_actions.shape}')
        tf.debugging.check_numerics(log_prob_actions, message=f'log_prob_actions is not numeric, {log_prob_actions}')
        log_prob_actions = tf.reduce_sum(log_prob_actions, axis=1, keepdims=True)
        # print(f'in implicit actor, in call, log_prob_actions: {log_prob_actions.shape}')
        # tf.debugging.check_numerics(log_prob_actions, message=f'log_prob_actions is not numeric, {log_prob_actions}')
        log_prob_actions = tf.reshape(log_prob_actions, shape=(state.shape[0], self.action_num))
        # print(f'in implicit actor, in call, log_prob_actions: {log_prob_actions.shape}')
        # tf.debugging.check_numerics(log_prob_actions, message=f'log_prob_actions is not numeric, {log_prob_actions}')
        prob_actions = tf.reduce_sum(tf.exp(log_prob_actions), axis=1, keepdims=True)
        # print(f'in implicit actor, in call, prob_actions: {prob_actions.shape}')
        # tf.debugging.check_numerics(prob_actions, message=f'prob_actions is not numeric, {prob_actions}')

        log_prob = tf.math.log(tf.divide((prob_main + prob_actions + 1e-6) , (self.action_num + 1)))
        # print(f'in implicit actor, in call, log_prob: {log_prob.shape}')
        tf.debugging.check_numerics(log_prob, message=f'log_prob is not numeric, {log_prob}')

        return action, log_prob

    def sample(self, state):
        noise = tf.random.normal(shape=(state.shape[0], self.noise_dim), mean=0, stddev=1)
        # print(f'in implicit actor, in sample, noise: {noise.shape}')

        l1 = self.l1(tf.concat([state, noise], axis=1))
        # print(f'in implicit actor, in sample, l1: {l1.shape}')
        l2 = self.l2(l1)
        # print(f'in implicit actor, in sample, l2: {l2.shape}')
        mu = self.mu(l2)
        # print(f'in implicit actor, in sample, mu: {mu.shape}')
        std = self.mu(l2)
        # print(f'in implicit actor, in sample, std: {std.shape}')
        std = tf.squeeze(tf.exp(tf.clip_by_value(std[..., tf.newaxis], self.log_sig_min, self.log_sig_max)), axis=2)
        # print(f'in implicit actor, in sample, std: {std.shape}')

        dist = tfp.distributions.TransformedDistribution(
            tfp.distributions.Normal(loc=mu, scale=std),
            bijector = self.bijector
            )
        action = tf.squeeze(dist.sample())
        # print(f'in implicit actor, action: {action.shape}')

        return action

class DistCritic(Model):
    """
        Implicit Distributional Critic
    """
    def __init__(self,
                 quantile_num,
                 noise_dim,
                 obs_space,
                 action_space,
                 ):
        super(DistCritic,self,).__init__()
        self.quantile_num = quantile_num
        self.noise_dim = noise_dim

        self.obs_space = obs_space
        self.action_space = action_space

        self.initializer = initializers.he_normal()
        self.regularizer = regularizers.l2(l=0.005)

        self.l1 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2 = Dense(256, activation = 'relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.value = Dense(1, activation=None)

    def call(self, state, action):
        noise = tf.random.normal(shape=(state.shape[0], self.quantile_num*self.noise_dim), mean=0, stddev=1)
        # print(f'in Dist Critic, noise: {noise.shape}')
        state_repeat = tf.repeat(state, self.quantile_num, axis=0)
        # print(f'in Dist Critic, state_repeat: {state_repeat.shape}')
        action_repeat = tf.repeat(action, self.quantile_num, axis=0)
        # print(f'in Dist Critic, action_repeat: {action_repeat.shape}')
        noise_repeat = tf.reshape(noise, shape=(state.shape[0]*self.quantile_num, self.noise_dim))
        # print(f'in Dist Critic, noise_repeat: {noise_repeat.shape}')

        l1 = self.l1(tf.concat([state_repeat, action_repeat, noise_repeat], axis=1))
        # print(f'in Dist Critic, l1: {l1.shape}')
        l2 = self.l2(l1)
        # print(f'in Dist Critic, l2: {l2.shape}')
        value = self.value(l2)
        # print(f'in Dist Critic, value: {value.shape}')
        value = tf.reshape(value, shape=(state.shape[0], self.quantile_num))
        # print(f'in Dist Critic, value: {value.shape}')
        g_values = tf.sort(value, axis=1, direction='ASCENDING')
        # print(f'in Dist Critic, g_values: {g_values.shape}')

        return g_values


class Agent:
    """
    Argument:
        agent_config: agent configuration which is realted with RL algorithm => DDPG
            agent_config:
                {
                    name, gamma, tau, update_freq, batch_size, warm_up, lr_actor, lr_critic,
                    buffer_size, use_PER, use_ERE, reward_normalize,
                    alpha, qualtile_dim, noise_dim
                    extension = {
                        use_implicit_actor, use_automatic_entropy_tuning,
                        implicit_actor_config: {
                            action_num, noise_dim, log_sig_min, log_sig_max, log_prob_min, log_prob_max
                        },
                        gaussian_actor_config: {
                            noise_dim, log_sig_min, log_sig_max, log_prob_min, log_prob_max
                        }
                        automatic_alpha_config: {
                            use_target_entropy, target_entropy
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
        self.quantile_num = self.agent_config['quantile_num']
        self.noise_dim = self.agent_config['noise_dim']

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

        self.critic_main_1 = DistCritic(self.quantile_num, self.noise_dim, self.obs_space, self.act_space)
        self.critic_main_2 = DistCritic(self.quantile_num, self.noise_dim, self.obs_space, self.act_space)
        self.critic_target_1 = DistCritic(self.quantile_num, self.noise_dim, self.obs_space, self.act_space)
        self.critic_target_2 = DistCritic(self.quantile_num, self.noise_dim, self.obs_space, self.act_space)

        self.critic_target_1.set_weights(self.critic_main_1.get_weights())
        self.critic_target_2.set_weights(self.critic_main_2.get_weights())
        self.critic_opt_main_1 = Adam(self.critic_lr_main)
        self.critic_opt_main_2 = Adam(self.critic_lr_main)
        self.critic_main_1.compile(optimizer=self.critic_opt_main_1)
        self.critic_main_2.compile(optimizer=self.critic_opt_main_2)

        # extension config
        self.extension_config = self.agent_config['extension']

        if self.extension_config['use_implicit_actor']:
            self.actor_main     = ImplicitActor(self.extension_config['implicit_actor_config'], self.obs_space, self.act_space)
            self.actor_target   = ImplicitActor(self.extension_config['implicit_actor_config'], self.obs_space, self.act_space)
            self.actor_target.set_weights(self.actor_main.get_weights())
            self.actor_opt_main = Adam(self.actor_lr_main)
            self.actor_main.compile(optimizer=self.actor_opt_main)
        else:
            self.actor_main     = GaussianActor(self.extension_config['gaussian_actor_config'], self.obs_space, self.act_space)
            self.actor_target   = GaussianActor(self.extension_config['gaussian_actor_config'], self.obs_space, self.act_space)
            self.actor_target.set_weights(self.actor_main.get_weights())
            self.actor_opt_main = Adam(self.actor_lr_main)
            self.actor_main.compile(optimizer=self.actor_opt_main)

        if self.extension_config['use_automatic_entropy_tuning']:
            if self.extension_config['use_automatic_entropy_tuning']['use_target_entropy']:
                self.target_entropy = self.extension_config['use_automatic_entropy_tuning']['target_entropy']
            else:
                self.target_entropy = -np.prod(self.act_space).item()
            self.log_alpha = tf.Variable(0, trainable=True, dtype=tf.float32)
            self.alpha_optimizer = Adam(self.actor_lr_main)
        else:
            self.alpha = self.agent_config['alpha']

    def action(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        # print(f'in action, obs: {obs.shape}')
        raw_action = self.actor_main.sample(obs)
        # print(f'in action, raw_action: {raw_action.shape}')
        action = np.clip(raw_action.numpy(), -1, 1)
        # print(f'in action, clipped action: {action}')

        return action

    def sample_tau(self):
        tau_raw = tf.random.uniform(shape=(self.batch_size, self.quantile_num), dtype=tf.float32) + 0.1
        # print(f'tau_raw: {tau_raw.shape}')
        tau_raw = tf.divide(tau_raw ,tf.reduce_sum(tau_raw, axis=1, keepdims=True))
        # print(f'tau_raw: {tau_raw.shape}')

        tau_hat_raw = tf.cumsum(tau_raw, axis=1).numpy()
        # print(f'tau_hat_raw: {tau_hat_raw.shape}')
        tau_hat_raw_left  = tf.concat([tf.zeros(shape=(self.batch_size, 1)), tau_hat_raw], axis=1)
        # print(f'tau_hat_raw_left: {tau_hat_raw_left.shape}')
        tau_hat_raw_right = tf.concat([tau_hat_raw, tf.zeros(shape=(self.batch_size, 1))], axis=1)
        # print(f'tau_hat_raw_right: {tau_hat_raw_right.shape}')

        tau_hat = tf.nest.map_structure(lambda x, y: (x + y) / 2, tau_hat_raw_left, tau_hat_raw_right)
        # print(f'tau_hat: {tau_hat.shape}')
        tau_hat = tf.slice(tau_hat, [0,0], [self.batch_size, self.quantile_num])
        # print(f'tau_hat: {tau_hat.shape}')

        return tau_hat

        # tau_hat = np.zeros_like(tau_hat_raw)
        # tau_hat[:, 0:1] = tau_hat_raw[:, 0:1] / 2
        # tau_hat[:, 1:]  = (tau_hat_raw[:, 1:] + tau_hat_raw[:, :-1])/2

        # return tf.convert_to_tensor(tau_hat, dtype=tf.float32)

    def update_target(self):
        actor_weithgs = []
        actor_targets = self.actor_target.get_weights()
        
        for idx, weight in enumerate(self.actor_main.get_weights()):
            actor_weithgs.append(weight * self.tau + actor_targets[idx] * (1 - self.tau))
        self.actor_target.set_weights(actor_weithgs)

        critic_1_weithgs = []
        critic_targets_1 = self.critic_target_1.get_weights()
        
        for idx, weight in enumerate(self.critic_main_1.get_weights()):
            critic_1_weithgs.append(weight * self.tau + critic_targets_1[idx] * (1 - self.tau))
        self.critic_target_1.set_weights(critic_1_weithgs)
        
        critic_2_weithgs = []
        critic_targets_2 = self.critic_target_2.get_weights()
        
        for idx, weight in enumerate(self.critic_main_2.get_weights()):
            critic_2_weithgs.append(weight * self.tau + critic_targets_2[idx] * (1 - self.tau))
        self.critic_target_2.set_weights(critic_2_weithgs)

    def update(self):
        if self.replay_buffer._len() < self.batch_size:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if not self.update_step % self.update_freq == 0:  # only update every update_freq
            self.update_step += 1
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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
            print(f'states : {states.shape}')
            print(f'next_states : {next_states.shape}')
            print(f'rewards : {rewards.shape}')
            print(f'actions : {actions.shape}')
            print(f'dones : {dones.shape}')
            print(f'is_weight : {is_weight.shape}')

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
            # print(f'states : {states.shape}')
            # print(f'next_states : {next_states.shape}')
            # print(f'rewards : {rewards.shape}')
            # print(f'actions : {actions.shape}')
            # print(f'dones : {dones.shape}')
            
        if self.extension_config['use_automatic_entropy_tuning']:
            alpha_variable = self.log_alpha
            with tf.GradientTape() as tape_alpha:
                tape_alpha.watch(alpha_variable)
                target_actions, taget_log_p = self.actor_target(next_states)
                print(f'target_actions : {target_actions.shape}')
                print(f'taget_log_p : {taget_log_p.shape}')
                alpha_loss = -tf.reduce_mean((tf.exp(self.log_alpha) * (taget_log_p + self.target_entropy)))
                print(f'alpha_loss : {alpha_loss.shape}')

            grads_alpha, _ = tf.clip_by_global_norm(tape_alpha.gradient(alpha_loss, alpha_variable), 0.5)
            self.alpha_optimizer.apply_gradients(zip(grads_alpha, alpha_variable))
   
        else:
            alpha_loss = 0
            alpha = self.alpha 

        critic_variable_1 = self.critic_main_1.trainable_variables
        critic_variable_2 = self.critic_main_2.trainable_variables
        with tf.GradientTape() as tape_critic_1, tf.GradientTape() as tape_critic_2:
            tape_critic_1.watch(critic_variable_1)
            tape_critic_2.watch(critic_variable_2)
            
            target_actions, target_log_p = self.actor_target(next_states)
            # print(f'target_actions : {target_actions.shape}')
            # print(f'target_log_p : {target_log_p.shape}')
            # tf.debugging.check_numerics(target_actions, message='target_actions is not numeric')
            # tf.debugging.check_numerics(target_log_p, message='target_log_p is not numeric')

            target_q_1_values = self.critic_target_1(next_states, target_actions)
            target_q_2_values = self.critic_target_2(next_states, target_actions)
            target_q_next = tf.stop_gradient(tf.minimum(target_q_1_values, target_q_2_values) - tf.multiply(alpha, target_log_p))
            # print(f'target_q_values : {target_q_1_values.shape}, {target_q_2_values.shape}')
            # print(f'target_q_next : {target_q_next.shape}')
            # tf.debugging.check_numerics(target_q_1_values, message='target_q_1_values is not numeric')
            # tf.debugging.check_numerics(target_q_2_values, message='target_q_2_values is not numeric')
            # tf.debugging.check_numerics(target_q_next, message='target_q_next is not numeric')

            target_q = tf.expand_dims(rewards, axis=1) + self.gamma * target_q_next * tf.expand_dims((1.0 - tf.cast(dones, dtype=tf.float32)), axis=1)
            # print(f'target_q : {target_q.shape}')
            # tf.debugging.check_numerics(target_q, message='target_q is not numeric')

            tau_1_hat = self.sample_tau()
            tau_2_hat = self.sample_tau()
            # print(f'tau_hat : {tau_1_hat.shape}, {tau_2_hat.shape}')
            # tf.debugging.check_numerics(tau_1_hat, message='tau_1_hat is not numeric')
            # tf.debugging.check_numerics(tau_2_hat, message='tau_2_hat is not numeric')

            current_q_1_values = self.critic_main_1(states, actions)
            current_q_2_values = self.critic_main_2(states, actions)
            # print(f'current_q_values : {current_q_1_values.shape}, {current_q_2_values.shape}')
            # tf.debugging.check_numerics(current_q_1_values, message='current_q_1_values is not numeric')
            # tf.debugging.check_numerics(current_q_2_values, message='current_q_2_values is not numeric')

            current_1_tile = tf.tile(tf.expand_dims(current_q_1_values, axis=2), [1, 1, self.quantile_num])
            current_2_tile = tf.tile(tf.expand_dims(current_q_2_values, axis=2), [1, 1, self.quantile_num])
            # print(f'in QR Loss, current_tile: {current_1_tile.shape}, {current_2_tile.shape}')
            # tf.debugging.check_numerics(current_1_tile, message='current_1_tile is not numeric')
            # tf.debugging.check_numerics(current_2_tile, message='current_2_tile is not numeric')

            target_tile = tf.tile(tf.expand_dims(target_q, axis=1), [1, self.quantile_num, 1])
            # print(f'in QR Loss, target_tile: {target_tile.shape}')
            # tf.debugging.check_numerics(target_tile, message='target_tile is not numeric')

            tau_1_hat_expand = tf.expand_dims(tau_1_hat, axis=2)
            tau_2_hat_expand = tf.expand_dims(tau_2_hat, axis=2)
            inv_tau_1 = tf.subtract(tf.ones_like(tau_1_hat_expand, dtype=tf.float32), tau_1_hat_expand)
            inv_tau_2 = tf.subtract(tf.ones_like(tau_2_hat_expand, dtype=tf.float32), tau_2_hat_expand)
            # print(f'in QR Loss, tau_hat_expand: {tau_1_hat_expand.shape}, {tau_2_hat_expand.shape}')
            # print(f'in QR Loss, inv_tau: {inv_tau_1.shape}, {inv_tau_2.shape}')

            td_loss_1 = tf.subtract(target_tile, current_1_tile)
            td_loss_2 = tf.subtract(target_tile, current_2_tile)
            # print(f'in QR Loss, td_loss: {td_loss_1.shape}, {td_loss_2.shape}')
            # tf.debugging.check_numerics(td_loss_1, message='td_loss_1 is not numeric')
            # tf.debugging.check_numerics(td_loss_2, message='td_loss_2 is not numeric')

            huber_loss_1 = tf.where(tf.less(td_loss_1, 1.0), 1/2 * tf.math.square(td_loss_1), 1.0 * tf.abs(td_loss_1 - 1.0 * 1/2))
            huber_loss_2 = tf.where(tf.less(td_loss_2, 1.0), 1/2 * tf.math.square(td_loss_2), 1.0 * tf.abs(td_loss_2 - 1.0 * 1/2))
            # print(f'in QR Loss, huber_loss: {huber_loss_1.shape}, {huber_loss_2.shape}')
            # tf.debugging.check_numerics(huber_loss_1, message='huber_loss_1 is not numeric')
            # tf.debugging.check_numerics(huber_loss_2, message='huber_loss_2 is not numeric')

            sign_1 = tf.sign(target_tile - current_1_tile)
            sign_2 = tf.sign(target_tile - current_2_tile)
            # print(f'in QR Loss, sign: {sign_1.shape}, {sign_2.shape}')
            # tf.debugging.check_numerics(sign_1, message='sign_1 is not numeric')
            # tf.debugging.check_numerics(sign_2, message='sign_2 is not numeric')

            rho_1 = tf.where(tf.less(sign_1, 0.0), tf.multiply(inv_tau_1 , huber_loss_1), tf.multiply(tau_1_hat_expand , huber_loss_1))
            rho_2 = tf.where(tf.less(sign_2, 0.0), tf.multiply(inv_tau_2 , huber_loss_2), tf.multiply(tau_2_hat_expand , huber_loss_2))
            # print(f'rho : {rho_1.shape}, {rho_2.shape}')
            # tf.debugging.check_numerics(rho_1, message='rho_1 is not numeric')
            # tf.debugging.check_numerics(rho_2, message='rho_2 is not numeric')

            td_error_1 = tf.reduce_mean(tf.reduce_sum(rho_1, axis = 2), axis=1)
            td_error_2 = tf.reduce_mean(tf.reduce_sum(rho_2, axis = 2), axis=1)
            # print(f'td_error : {td_error_1.shape}, {td_error_2.shape}')
            # tf.debugging.check_numerics(td_error_1, message='td_error_1 is not numeric')
            # tf.debugging.check_numerics(td_error_2, message='td_error_2 is not numeric')

            critic_losses_1 = tf.cond(tf.convert_to_tensor(self.agent_config['use_PER'], dtype=tf.bool), \
                    lambda: tf.multiply(is_weight, tf.math.square(td_error_1)), \
                    lambda: tf.math.square(td_error_1))
            critic_losses_2 = tf.cond(tf.convert_to_tensor(self.agent_config['use_PER'], dtype=tf.bool), \
                    lambda: tf.multiply(is_weight, tf.math.square(td_error_2)), \
                    lambda: tf.math.square(td_error_2))
            # print(f'critic_losses : {critic_losses_1.shape}, {critic_losses_2.shape}')
            # tf.debugging.check_numerics(critic_losses_1, message='critic_losses_1 is not numeric')
            # tf.debugging.check_numerics(critic_losses_2, message='critic_losses_2 is not numeric')

            critic_loss_1 = tf.math.reduce_mean(critic_losses_1)
            critic_loss_2 = tf.math.reduce_mean(critic_losses_2)
            # print(f'critic_loss : {critic_loss_1.shape}, {critic_loss_2.shape}')
            # tf.debugging.check_numerics(critic_loss_1, message='critic_loss_1 is not numeric')
            # tf.debugging.check_numerics(critic_loss_2, message='critic_loss_2 is not numeric')

        grads_critic_1, _ = tf.clip_by_global_norm(tape_critic_1.gradient(critic_loss_1, critic_variable_1), 0.5)
        grads_critic_2, _ = tf.clip_by_global_norm(tape_critic_2.gradient(critic_loss_2, critic_variable_2), 0.5)

        self.critic_opt_main_1.apply_gradients(zip(grads_critic_1, critic_variable_1))
        self.critic_opt_main_2.apply_gradients(zip(grads_critic_2, critic_variable_2))

        actor_variable = self.actor_main.trainable_variables       
        with tf.GradientTape() as tape_actor:
            tape_actor.watch(actor_variable)

            new_policy_actions, new_log_p = self.actor_main(states)
            # print(f'new_policy_actions : {new_policy_actions.shape}')
            # print(f'new_log_p : {new_log_p.shape}')
            # tf.debugging.check_numerics(new_policy_actions, message='new_policy_actions is not numeric')
            # tf.debugging.check_numerics(new_log_p, message='new_log_p is not numeric')

            new_current_q_1 = self.critic_main_1(states, new_policy_actions)
            new_current_q_2 = self.critic_main_2(states, new_policy_actions)
            # print(f'new_current_q_1 : {new_current_q_1.shape}')
            # print(f'new_current_q_2 : {new_current_q_2.shape}')
            # tf.debugging.check_numerics(new_current_q_1, message='new_current_q_1 is not numeric')
            # tf.debugging.check_numerics(new_current_q_2, message='new_current_q_2 is not numeric')

            new_current_q = tf.multiply(alpha, new_log_p) - tf.minimum(new_current_q_1, new_current_q_2)
            # print(f'new_current_q : {new_current_q.shape}')
            # tf.debugging.check_numerics(new_current_q_2, message='new_current_q_2 is not numeric')

            actor_losses = tf.reduce_mean(new_current_q, axis=1)
            # print(f'actor_losses : {actor_losses.shape}')
            # tf.debugging.check_numerics(actor_losses, message='actor_losses is not numeric')
            actor_loss = tf.reduce_mean(actor_losses)
            # print(f'actor_loss : {actor_loss.shape}')
            # tf.debugging.check_numerics(actor_loss, message='actor_loss is not numeric')
            
        grads_actor, _ = tf.clip_by_global_norm(tape_actor.gradient(actor_loss, actor_variable), 0.5)
        self.actor_opt_main.apply_gradients(zip(grads_actor, actor_variable))

        target_q_val = tf.reduce_mean(target_q).numpy()
        current_q_1_val = tf.reduce_mean(current_q_1_values).numpy()
        current_q_2_val = tf.reduce_mean(current_q_2_values).numpy()
        criitic_loss_1_val = critic_loss_1.numpy()
        criitic_loss_2_val = critic_loss_2.numpy()
        if self.extension_config['use_automatic_entropy_tuning']:
            alpha_loss_val = alpha_loss.numpy()
        else:
            alpha_loss_val = alpha_loss
        actor_loss_val = actor_loss.numpy()
        
        self.update_target()

        td_error_numpy = np.abs((tf.add(td_error_1, td_error_2)/2).numpy())
        if self.agent_config['use_PER']:
            for i in range(self.batch_size):
                self.replay_buffer.update(idxs[i], td_error_numpy[i])

        return updated, alpha_loss_val, actor_loss_val, criitic_loss_1_val, criitic_loss_2_val, target_q_val, current_q_1_val, current_q_2_val

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