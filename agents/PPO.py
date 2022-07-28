from collections import defaultdict
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
from utils.replay_buffer_SIL import SILExperienceMemory
from copy import deepcopy

tf.executing_eagerly()


class Actor(Model):
    def __init__(self, obs_space, action_space):
        super(Actor,self).__init__()
        self.initializer = initializers.orthogonal()
        self.regularizer = regularizers.l2(l=0.0005)

        self.l1 = Dense(256, activation = 'swish', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l1_ln = LayerNormalization(axis=-1)
        self.l2 = Dense(256, activation = 'swish', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_ln = LayerNormalization(axis=-1)
        self.l3 = Dense(64, activation = 'swish', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3_ln = LayerNormalization(axis=-1)
        self.mu = Dense(action_space, activation='tanh')
        self.log_std = Dense(action_space, activation='tanh')

    def call(self, state):
        l1 = self.l1_ln(self.l1(state))
        l2 = self.l2_ln(self.l2(l1))
        l3 = self.l3_ln(self.l3(l2))
        mu = self.mu(l3)
        log_std = self.log_std(l3)

        return mu, log_std


class Critic(Model):
    def __init__(self):
        super(Critic,self).__init__()
        self.initializer = initializers.orthogonal()
        self.regularizer = regularizers.l2(l=0.0005)
        
        self.l1 = Dense(256, activation = 'swish' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l1_ln = LayerNormalization(axis=-1)
        self.l2 = Dense(256, activation = 'swish' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_ln = LayerNormalization(axis=-1)
        self.l3 = Dense(64, activation = 'swish' , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3_ln = LayerNormalization(axis=-1)
        self.value = Dense(1, activation = None)

    def call(self, state):
        l1 = self.l1_ln(self.l1(state))
        l2 = self.l2_ln(self.l2(l1))
        l3 = self.l3_ln(self.l3(l2))
        value = self.value(l3)

        return value


class Agent:
    """
    Argument:
        agent_config: agent configuration which is realted with RL algorithm => PPO(Vanilla, SIL)
            agent_config:
                {
                    name, gamma, total_batch_size, batch_size, epoch_num,
                    entropy_coeff, entropy_coeff_reduction_rate, entropy_coeff_min,
                    epsilon, std_bound, lr_actor, lr_critic, reward_normalize,
                    reward_min, reward_max, log_prob_min, log_prob_max

                    extension = {
                        name, use_GAE, use_SIL,

                        GAE_config = {
                            use_gae_norm, lambda
                        },
                        SIL_config = {
                            buffer_size, batch_size, lr_sil, return_criteria, epoch_num, log_prob_min, log_prob_max
                        }
                    }
                }
        obs_shape_n: shpae of observation
        act_shape_n: shape of action

    Methods:
        action: return the action which is mapped with obs in policy
        get_gaes: return the values of general advantage estimator(GAE) and target values
        update: update main critic/actor network via PPO
        self_imitation_learning: update main critic/actor network via SIL
        save_xp: save transition(s, a, r, s', d) in experience memory, and transmit the trajectory to update_buffer method
        update_buffer: save trajectory which has higher return value than return_criteria
        add_episode: save trajectory directly in sil_buffer
        discount_with_dones: calculate the returns in trajectory using rewards
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

        self.update_step = 0

        self.total_batch_size = self.agent_config['total_batch_size']
        self.replay_buffer = ExperienceMemory(self.total_batch_size)
        self.batch_size = self.agent_config['batch_size']
        self.epoch = self.agent_config['epoch_num']

        self.entropy_coeff = self.agent_config['entropy_coeff']
        self.entropy_coeff_reduction_rate = self.agent_config['entropy_coeff_reduction_rate']
        self.entropy_coeff_min = self.agent_config['entropy_coeff_min']
        self.epsilon = self.agent_config['epsilon']

        self.std_bound  = self.agent_config['std_bound']
        self.std_min    = self.agent_config['std_min']
        self.std_reduction_rate = self.agent_config['std_reduction_rate']

        self.reward_min = self.agent_config['reward_min']
        self.reward_max = self.agent_config['reward_max']

        self.log_prob_min = self.agent_config['log_prob_min']
        self.log_prob_max = self.agent_config['log_prob_max']

        # network config
        self.actor_lr = self.agent_config['lr_actor']
        self.critic_lr = self.agent_config['lr_critic']

        self.actor = Actor(self.obs_space, self.act_space)
        self.actor_opt = Adam(self.actor_lr)
        # self.actor.compile(optimizer=self.actor_opt)
        
        self.critic = Critic()
        self.critic_opt = Adam(self.critic_lr)
        # self.critic.compile(optimizer=self.critic_opt)

        # extension config
        self.extension_config = self.agent_config['extension']
        self.extension_name = self.extension_config['name']

        if self.extension_config['use_GAE']:
            self.use_gae_norm = self.extension_config['GAE_config']['use_gae_norm']
            self.lamda = self.extension_config['GAE_config']['lambda']
        
        if self.extension_config['use_SIL']:
            self.sil_config = self.extension_config['SIL_config']

            self.trajectory = defaultdict(lambda:[])
            self.sil_update_step = 0

            self.sil_buffer = SILExperienceMemory(self.sil_config['buffer_size'], 0.6, 0.1)
            self.sil_batch_size = self.sil_config['batch_size']
            self.sil_min_batch_size = self.sil_config['min_batch_size']
            self.sil_epoch = self.sil_config['epoch_num']

            self.sil_log_prob_min = self.sil_config['log_prob_min']
            self.sil_log_prob_max = self.sil_config['log_prob_min']

            self.sil_adv_min = self.sil_config['adv_min']
            self.sil_adv_max = self.sil_config['adv_max']

            self.sil_lr_actor = self.sil_config['lr_sil_actor']
            self.sil_actor_opt = Adam(learning_rate=self.sil_lr_actor)

            self.sil_lr_critic = self.sil_config['lr_sil_critic']
            self.sil_critic_opt = Adam(learning_rate=self.sil_lr_critic)

            self.return_criteria = self.sil_config['return_criteria']
            self.naive_criteria = self.sil_config['naive_criteria']
            self.recent_return_coeff = self.sil_config['recent_return_coeff']

    def action(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)

        mu, log_std = self.actor(obs)
        std = tf.clip_by_value(tf.exp(log_std), clip_value_min=self.std_bound[0], clip_value_max=self.std_bound[1])
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        action = dist.sample()
        # print(f'in action, obs : {obs.shape}')
        # print(f'mu, std_ : {mu.shape}, {std_.shape}')
        # print(f'action : {action.shape}')

        log_prob = tf.clip_by_value(dist.log_prob(action), self.log_prob_min, self.log_prob_max)
        # print(f'log_prob : {log_prob.shape}')
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=False)
        # print(f'log_prob : {log_prob.shape}')

        action = tf.squeeze(action).numpy()
        action = np.clip(action, -1, 1)

        return action, log_prob.numpy()[0]
    
    def get_gaes(self, rewards, dones, values, next_values):
        deltas = [r + self.gamma * (1 - d) * nv - v for r, d, v, nv in zip(rewards, dones, values, next_values)]
        deltas = np.stack(deltas)
        gaes = deepcopy(deltas)

        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * self.gamma * self.lamda * gaes[t + 1]
        
        gaes_norm = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        if self.use_gae_norm:
            gaes = gaes_norm

        target = gaes + values

        return gaes, target

    def update(self):
        if self.replay_buffer._len() < self.total_batch_size:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        self.update_step += 1

        raw_states, raw_next_states, raw_rewards, raw_actions, raw_old_log_policies, raw_dones = self.replay_buffer.sample()
        self.replay_buffer.clear()

        raw_states      = np.array(raw_states, dtype=np.float32)
        raw_next_states = np.array(raw_next_states, dtype=np.float32)
        raw_rewards     = np.array(raw_rewards, dtype=np.float32)
        raw_actions     = np.array(raw_actions, dtype=np.float32)
        raw_old_log_policies   = np.array(raw_old_log_policies, dtype=np.float32)
        raw_dones       = np.array(raw_dones, dtype=np.float32)

        raw_values      = self.critic(tf.convert_to_tensor(raw_states, dtype=tf.float32))
        raw_values      = np.array(raw_values)

        raw_next_values = self.critic(tf.convert_to_tensor(raw_next_states, dtype=tf.float32))
        raw_next_values = np.array(raw_next_values)

        if self.agent_config['reward_normalize']:
            # raw_rewards = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-5)
            raw_rewards = np.clip(raw_rewards, self.reward_min, self.reward_max)

        raw_advs, raw_targets = self.get_gaes(rewards=raw_rewards,
                                      dones=raw_dones,
                                      values=raw_values,
                                      next_values=raw_next_values)

        # for logging
        std_mem         = 0
        entropy_mem     = 0
        ratio_mem       = 0
        actor_loss_mem  = 0
        adv_mem         = 0
        target_val_mem  = 0
        current_val_mem = 0
        critic_loss_mem = 0

        for _ in range(self.epoch):
            sample_range = np.arange(self.total_batch_size)
            np.random.shuffle(sample_range)
            sample_idx = sample_range[:self.batch_size]
            
            batch_states = raw_states[sample_idx]
            batch_actions = raw_actions[sample_idx]
            batch_targets = raw_targets[sample_idx]
            batch_old_log_policies = raw_old_log_policies[sample_idx]
            batch_advs = raw_advs[sample_idx]

            critic_variable = self.critic.trainable_variables
            with tf.GradientTape() as tape_critic:
                tape_critic.watch(critic_variable)
                current_value = tf.squeeze(self.critic(tf.convert_to_tensor(batch_states, dtype=tf.float32)))
                # print(f'current_value : {current_value.shape}')

                target_value = tf.squeeze(tf.convert_to_tensor(batch_targets, dtype=tf.float32))
                # print(f'target_value : {target_value.shape}')

                td_error = tf.subtract(current_value, target_value)
                # print(f'td_error : {td_error.shape}')
                huber_loss = tf.where(tf.less(td_error, 1.0), 1/2 * tf.math.square(td_error), 1.0 * tf.abs(td_error - 1.0 * 1/2))

                critic_loss = tf.math.reduce_mean(huber_loss)
                # print(f'critic_loss : {critic_loss.shape}')

            grads_critic, _ = tf.clip_by_global_norm(tape_critic.gradient(critic_loss, critic_variable), 0.5)

            self.critic_opt.apply_gradients(zip(grads_critic, critic_variable))

            actor_variable = self.actor.trainable_variables   
            with tf.GradientTape() as tape:
                tape.watch(actor_variable)
                mu, log_std = self.actor(tf.convert_to_tensor(batch_states, dtype=tf.float32))
                std = tf.exp(log_std)
                # print(f'mu : {mu.shape}, std : {std.shape}')

                dist = tfp.distributions.Normal(loc = mu, scale = tf.clip_by_value(std, clip_value_min=self.std_bound[0], clip_value_max=self.std_bound[1]))

                entropy = tf.reduce_mean(dist.entropy())
                # print(f'entropy : {entropy.shape}')

                actions = tf.convert_to_tensor(batch_actions, dtype=tf.float32)
                # print(f'actions : {actions.shape}')

                log_policy = tf.clip_by_value(dist.log_prob(actions), self.log_prob_min, self.log_prob_max)
                # print(f'log_policy : {log_policy.shape}')
                log_policy = tf.reduce_sum(log_policy, 1, keepdims=False)
                # print(f'log_policy : {log_policy.shape}')

                old_log_policy = tf.convert_to_tensor(batch_old_log_policies, dtype=tf.float32)
                # print(f'old_log_policy : {old_log_policy.shape}')

                adv = tf.squeeze(tf.convert_to_tensor(batch_advs, dtype=tf.float32))
                # print(f'adv : {adv.shape}')

                ratio = tf.exp(log_policy - old_log_policy)
                # print(f'ratio : {ratio.shape}')

                surr1 = tf.multiply(adv, ratio)
                # print(f'surr1 : {surr1.shape}')
                # surr2 = tf.multiply(adv, tf.clip_by_value_preserve_gradient(ratio, clip_value_min=1-self.epsilon, clip_value_max=1+self.epsilon)) # ambiguous whether the preserving the gradient of ratio
                surr2 = tf.multiply(adv, tf.clip_by_value(ratio, clip_value_min=1-self.epsilon, clip_value_max=1+self.epsilon))
                # print(f'surr2 : {surr2.shape}')

                minimum = tf.minimum(surr1, surr2)
                # print(f'minimum : {minimum.shape}')

                actor_loss = -tf.reduce_mean(minimum) - self.entropy_coeff * entropy
                # print(f'actor_loss : {actor_loss.shape}')

            grads, _ = tf.clip_by_global_norm(tape.gradient(actor_loss, actor_variable), 0.5)

            self.actor_opt.apply_gradients(zip(grads, actor_variable))

            advantage = tf.reduce_mean(adv).numpy()
            target_value = tf.reduce_mean(target_value).numpy()
            current_value = tf.reduce_mean(current_value).numpy()

            std_mem += tf.reduce_mean(std).numpy() / self.epoch
            entropy_mem += entropy.numpy() / self.epoch
            ratio_mem += tf.reduce_mean(ratio).numpy() / self.epoch
            actor_loss_mem += actor_loss.numpy() / self.epoch
            adv_mem += advantage / self.epoch
            target_val_mem += target_value / self.epoch
            current_val_mem += current_value / self.epoch
            critic_loss_mem += critic_loss.numpy() / self.epoch

        return True, std_mem, entropy_mem, ratio_mem, actor_loss_mem, adv_mem, target_val_mem, current_val_mem, critic_loss_mem

    def self_imitation_learning(self):
        if self.sil_buffer._len() < self.sil_batch_size:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        self.sil_update_step += 1

        # for logging
        entropy_mem     = 0
        actor_loss_mem  = 0
        adv_mem         = 0
        target_val_mem  = 0
        current_val_mem = 0
        critic_loss_mem = 0

        states, actions, returns, indices, weights = self.sil_buffer.sample(self.sil_batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        # print(f'states : {states.shape}')
        # print(f'returns : {returns.shape}')
        # print(f'weights : {weights.shape}')
        # print(f'actions : {actions.shape}')

        critic_variable = self.critic.trainable_variables
        actor_variable  = self.actor.trainable_variables
        with tf.GradientTape() as tape_sil_for_critic, tf.GradientTape() as tape_sil_for_actor:
            tape_sil_for_critic.watch(critic_variable)
            tape_sil_for_actor.watch(actor_variable)

            current_value = tf.squeeze(self.critic(states))
            # print(f'current_value : {current_value.shape}')

            mask = tf.where(returns - current_value > 0.0, tf.ones_like(returns), tf.zeros_like(returns)) # train_adv가 0보다 크면 1, 아니면 0으로 채워진 train_adv와 같은 형태의 mask 선언
            # print(f'mask : {mask.shape}')
            num_samples = tf.maximum(tf.reduce_sum(mask), self.sil_min_batch_size) # Todo
            # print(f'num_samples : {num_samples.shape}')

            # print('sil run')
            advs = tf.reduce_sum(tf.stop_gradient(tf.clip_by_value(returns - current_value, 0, self.sil_adv_max))) / num_samples # Todo
            # print(f'advs : {advs.shape}')
            delta = tf.stop_gradient(tf.multiply(tf.clip_by_value(current_value - returns, self.sil_adv_min, 0), mask))
            # print(f'delta : {delta.shape}')

            sil_value_error = tf.multiply(tf.multiply(delta, weights), current_value)

            critic_loss = tf.reduce_sum(sil_value_error) / num_samples
            # print(f'sil_value_error : {sil_value_error.shape}')
            # print(f'critic_loss : {critic_loss.shape}')

            mu, std = self.actor(states)
            # print(f'mu : {mu.shape}, std : {std.shape}')

            # dist = tfp.distributions.Normal(loc = mu, scale = tfp.math.clip_by_value_preserve_gradient(std, clip_value_min=self.std_bound[0], clip_value_max=self.std_bound[1]))
            dist = tfp.distributions.Normal(loc = mu, scale = tf.clip_by_value(std, clip_value_min=self.std_bound[0], clip_value_max=self.std_bound[1]))

            entropy_raw = tf.reduce_mean(dist.entropy(), 1, keepdims=False)
            entropy_weight = tf.multiply(tf.multiply(weights, entropy_raw), mask)
            entropy = tf.reduce_sum(entropy_weight) / num_samples # 배치별 entropy를 평균 내줌
            # print(f'entropy_raw : {entropy_raw.shape}')
            # print(f'entropy_weight : {entropy_weight.shape}')
            # print(f'entropy : {entropy.shape}')

            log_policy = -tf.reduce_mean(dist.log_prob(actions), 1, keepdims=False)
            # clipped_log_policy = tf.minimum(log_policy, self.sil_log_prob_max) # https://github.com/TianhongDai/self-imitation-learning-pytorch
            clipped_log_policy = tf.math.add(tf.stop_gradient(tf.minimum(log_policy, self.sil_log_prob_max) - log_policy), log_policy) # https://github.com/junhyukoh/self-imitation-learning

            # print(f'log_policy : {log_policy.shape}')
            # print(f'clipped_log_policy : {clipped_log_policy.shape}')

            actor_loss_raw = tf.multiply(tf.multiply(tf.multiply(advs, clipped_log_policy), weights), mask)
            actor_loss = tf.reduce_sum(actor_loss_raw) / num_samples
            actor_loss_entropy = actor_loss - self.entropy_coeff * entropy
            # print(f'actor_loss_raw : {actor_loss_raw.shape}')
            # print(f'actor_loss : {actor_loss.shape}')
            # print(f'actor_loss_entropy : {actor_loss_entropy.shape}')

        grads_critic, _ = tf.clip_by_global_norm(tape_sil_for_critic.gradient(critic_loss, critic_variable), 0.5)
        grads_actor, _ = tf.clip_by_global_norm(tape_sil_for_actor.gradient(actor_loss_entropy, actor_variable), 0.5)

        self.sil_critic_opt.apply_gradients(zip(grads_critic, critic_variable))
        self.sil_actor_opt.apply_gradients(zip(grads_actor, actor_variable))

        target_value = (tf.reduce_sum(tf.multiply(tf.multiply(weights, returns), mask)) / num_samples).numpy()
        current_value = (tf.reduce_sum(tf.multiply(tf.multiply(weights, current_value), mask)) / num_samples).numpy()

        advantages = np.squeeze(deepcopy(tf.multiply(advs, mask)).numpy())
        self.sil_buffer.update(indices, advantages)

        entropy_mem += entropy.numpy()
        actor_loss_mem += actor_loss.numpy()
        adv_mem += tf.divide(tf.reduce_sum(tf.multiply(advs, mask)), num_samples).numpy()
        target_val_mem += target_value
        current_val_mem += current_value
        critic_loss_mem += critic_loss.numpy()

        return True, entropy_mem, actor_loss_mem, adv_mem, target_val_mem, current_val_mem, critic_loss_mem

    def save_xp(self, state, next_state, reward, action, log_policy, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add((state, next_state, reward / 10, action, log_policy, done))

        # Store trajectory in the sil_replay buffer via update_buffer function
        if self.extension_config['use_SIL']:
            if done == False:
                self.trajectory['state'].append(state)
                self.trajectory['action'].append(action)
                self.trajectory['reward'].append(reward)
                self.trajectory['done'].append(done)

            else:
                self.update_buffer()

                for _, vals in self.trajectory.items():
                    vals.clear()


    def update_buffer(self):
        better_return = False
        
        r = sum(self.trajectory['reward'])

        if r > self.return_criteria:
            better_return = True
        
        if better_return:
            self.add_episode()

    def add_episode(self):
        states = self.trajectory['state']
        actions = self.trajectory['action']
        rewards = self.trajectory['reward']
        dones = self.trajectory['done']

        returns_g = self.discount_with_dones(rewards, dones, self.gamma)
        
        for (state, action, return_g) in list(zip(states, actions, returns_g)):
            self.sil_buffer.add((state, action, return_g))
            
    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0

        for reward, done in zip(rewards[::-1], dones[::-1]):
            if self.agent_config['reward_normalize']:
                reward = np.clip(reward, self.reward_min, self.reward_max)

            r = reward + gamma * r * (1. - done)
            discounted.append(r)

        discounted = np.array(discounted, dtype=np.float32)

        if discounted.std() >= 0:
            discounted_norm = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
        else:
            discounted_norm = (discounted - discounted.mean()) / (discounted.std() - 1e-8)        

        return discounted_norm[::-1]

    def load_models(self, path):
        print('Load Model Path : ', path)
        self.actor.load_weights(path, "_actor")
        self.critic.load_weights(path, "_critic")

    def save_models(self, path, score):
        save_path = str(path) + "score_" + str(score) + "_model"
        print('Save Model Path : ', save_path)
        self.actor.save_weights(save_path, "_actor")
        self.critic.save_weights(save_path, "_critic")

