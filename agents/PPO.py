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
        self.std = Dense(action_space, activation='softplus')

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

        self.update_step = 0

        self.replay_buffer = ExperienceMemory(2000)
        self.total_batch_size = self.agent_config['total_batch_size']
        self.batch_size = self.agent_config['batch_size']
        self.warm_up = self.agent_config['warm_up']
        self.epoch = self.agent_config['epoch_num']

        self.entropy_coeff = self.agent_config['entropy_coeff']
        self.entropy_reduction_rate = self.agent_config['entropy_reduction_rate']
        self.epsilon = self.agent_config['epsilon']

        self.std_bound = self.extension_config['std_bound']
        self.log_prob_min = self.agent_config['log_prob_min']
        self.log_prob_max = self.agent_config['log_prob_max']

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

        if self.extension_config['use_GAE']:
            self.use_gae_norm = self.extension_config['gae_config']['use_gae_norm']
            self.lamda = self.agent_config['lambda']
        
        if self.extension_config['use_SIL']:
            self.sil_config = self.extension_config['SIL_config']

            self.sil_update_step = 0

            self.sil_buffer = SILExperienceMemory(self.sil_config['buffer_size'], 0.6, 0.1)
            self.sil_batch_size = self.sil_config['batch_size']
            self.sil_epoch = self.sil_config['epoch']

            self.sil_lr = self.sil_config['lr_sil']
            self.sil_opt = Adam(learning_rate=self.sil_lr)

            self.sil_value_coeff = self.sil_config['value_coefficient']
            self.return_criteria = self.sil_config['return_criteria']

    def action(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)

        mu, std_ = self.actor(obs)
        std = tfp.math.clip_by_value_preserve_gradient(std_, clip_value_min=self.std_bound[0], clip_value_max=self.std_bound[1])
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        action = tf.squeeze(dist.sample())
        log_prob = tfp.math.clip_by_value_preserve_gradient(dist.log_prob(action)[..., tf.newaxis], self.log_prob_min, self.log_prob_max)
        log_policy = tf.reduce_sum(log_prob, axis=1, keepdims=False)

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

        if self.use_gae_norm:
            gaes = gaes_norm

        target = gaes + values

        return gaes, target

    def update(self):
        if self.replay_buffer._len() < self.total_batch_size:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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

        if self.agent_config['reward_normalize']:
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

                train_dist = tfp.distributions.Normal(loc = train_mu, scale = train_std)

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
                surr2 = tf.multiply(train_adv, tfp.math.clip_by_value_preserve_gradient(ratio, clip_value_min=1-self.epsilon, clip_value_max=1+self.epsilon))
                # surr2 = tf.multiply(train_adv, tf.clip_by_value(ratio, clip_value_min=1-self.epsilon, clip_value_max=1+self.epsilon))
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

        self.replay_buffer.clear()

        return True, entropy_mem, ratio_mem, actor_loss_mem, adv_mem, target_val_mem, current_val_mem, critic_loss_mem

    def self_imitation_learning(self):
        if self.sil_buffer._len() < self.total_batch_size:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            state, action, return_g, index, weight = self.sil_buffer.sample(self.sil_batch_size)

            critic_variable = self.critic.trainable_variables
            actor_variable  = self.actor.trainable_variables  
            with tf.GradientTape() as tape_sil_for_critic, tf.GradientTape() as tape_sil_for_actor:
                tape_sil_for_critic.watch(critic_variable)
                tape_sil_for_actor.watch(actor_variable)

                train_return_g = tf.convert_to_tensor(return_g, dtype=tf.float32)
                train_weight = tf.convert_to_tensor(weight, dtype=tf.float32)
                train_action = tf.convert_to_tensor(action, dtype=tf.float32)

                train_mu, train_std, train_current_value = self.ppo(tf.convert_to_tensor(state, dtype=tf.float32))
                train_current_value = tf.squeeze(train_current_value) ; # train_current_value = tf.reshape(train_current_value, [self.sil_batch_size,1])
                # train_std_ = tf.stop_gradient(tf.clip_by_value(train_std, clip_value_min=self.std_bound[0], clip_value_max=1))
                train_dist = tfp.distributions.Normal(loc = train_mu, scale = train_std)

                train_adv = train_return_g - train_current_value
                train_mask = tf.where(train_adv > 0.0, tf.ones_like(train_adv), tf.zeros_like(train_adv)) # train_adv가 0보다 크면 1, 아니면 0으로 채워진 train_adv와 같은 형태의 mask 선언
                train_num_samples = tf.reduce_sum(train_mask)

                if (train_num_samples.numpy() == 0):
                    print('sil pass because good samples 0')
                    return False, 0, 0, 0, 0, 0, 0

                # print('sil run')
                entropy_raw = tf.reduce_mean(train_dist.entropy(), 1, keepdims=False)
                entropy_weight = tf.multiply(tf.multiply(train_weight, entropy_raw), train_mask)
                entropy = tf.reduce_sum(entropy_weight) / train_num_samples # 배치별 entropy를 평균 내줌

                train_log_policy = tf.reduce_mean(train_dist.log_prob(train_action), 1, keepdims=False)
                train_clipped_log_policy = tf.stop_gradient(tf.clip_by_value(train_log_policy, clip_value_min=-10, clip_value_max=10))

                pi_loss_raw = tf.multiply(tf.multiply(tf.multiply(train_adv, train_clipped_log_policy),train_weight), train_mask)
                pi_loss = tf.reduce_sum(pi_loss_raw) / train_num_samples
                pi_loss_entropy = pi_loss - entropy * self.ppo_ent # entropy를 바로 사용하지 않고 entropy coefficent를 곱해줌

                train_sil_value_error = tf.multiply(tf.multiply(train_adv, train_weight), train_mask)
                value_loss = tf.reduce_sum(tf.square(train_sil_value_error)) / train_num_samples
                total_loss = pi_loss_entropy + self.ppo_sil_val * value_loss

                advantages = deepcopy(tf.multiply(train_adv, train_mask)) # 0 이하면 그냥 0으로 업데이트

            grads_critic, _ = tf.clip_by_global_norm(tape_sil_for_critic.gradient(critic_loss, critic_variable), 0.5)
            grads_actor, _ = tf.clip_by_global_norm(tape_sil_for_actor.gradient(actor_loss, actor_variable), 0.5)

            self.sil_opt.apply_gradients(zip(grads_critic, critic_variable))
            self.sil_opt.apply_gradients(zip(grads_actor, actor_variable))
            self.sil_update_step += 1

            value_target = (tf.reduce_sum(tf.multiply(tf.multiply(train_weight, train_return_g), train_mask)) / train_num_samples).numpy()
            value_cur = (tf.reduce_sum(tf.multiply(tf.multiply(train_weight, train_current_value), train_mask)) / train_num_samples).numpy()

            advantages = np.squeeze(advantages.numpy())
            self.sil_buffer.update(index, advantages)

        return True, entropy.numpy(), pi_loss_entropy.numpy(), value_target, value_cur, value_loss.numpy(), total_sil_loss.numpy()

    def update_buffer(self, trajectory):
        better_return = False
        
        for r in trajectory['reward']:
            if r > self.return_criteria:
                better_return = True
                break
        
        if better_return:
            self.add_episode(trajectory)

    def add_episode(self, trajectory):
        for key in trajectory.keys():
            if key == 'state':
                states = trajectory[key]

            elif key == 'reward':
                rewards = trajectory[key]

            elif key == 'action':
                actions = trajectory[key]

            elif key == 'done':
                dones = trajectory[key]

        returns_g = self.discount_with_dones(rewards, dones, self.gamma)
        
        for (state, action, return_g) in list(zip(states, actions, returns_g)):
            self.sil_buffer.add((state, action, return_g))
            
    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0

        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        
        return discounted[::-1]

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

