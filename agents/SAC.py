import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LayerNormalization

from utils.prioritized_memory_numpy import PrioritizedMemory
from utils.replay_buffer import ExperienceMemory


class Actor(Model):
    def __init__(self, action_space):
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

    def call(self, state):
        l1 = self.l1_ln(self.l1(state))
        l2 = self.l2_ln(self.l2(l1))
        l3 = self.l3_ln(self.l3(l2))
        l4 = self.l4_ln(self.l4(l3))
        mu = self.mu(l4)

        return mu

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
        self.value = tf.keras.layers.Dense(1, activation = None)

    def call(self, state_action):
        l1 = self.l1_ln(self.l1(state_action))
        l2 = self.l2_ln(self.l2(l1))
        l3 = self.l3_ln(self.l3(l2))
        l4 = self.l4_ln(self.l4(l3))
        value = self.value(l4)

        return value

class Agent:
    """
    input argument: obs_space, act_space, agent_config

    agent_config: agent_name, gamma, tau, update_freq, actor_update_freq, batch_size, warm_up,\
                  gaussian_std, noise_clip, noise_reduce_rate, lr_actor_main, lr_critic_main,\
                  use_PER, buffer_size, reward_normalize
    """
    def __init__(self, obs_space, act_space, agent_config):
        self.agent_config = agent_config
        self.name = self.agent_config['agent_name']

        self.obs_space = obs_space
        self.act_space = act_space
        print('obs_space: {}, act_space: {}'.format(self.obs_space, self.act_space))
 
        self.gamma = self.agent_config['gamma']
        self.tau = self.agent_config['tau']
        self.update_freq = self.agent_config['update_freq']
        self.actor_update_freq = self.agent_config['actor_update_freq'] # k번 critic update당 1번 policy update

        if self.agent_config['use_PER']:
            self.replay_buffer = PrioritizedMemory(self.agent_config['buffer_size'])
        else:
            self.replay_buffer = ExperienceMemory(self.agent_config['buffer_size'])
        self.batch_size = self.agent_config['batch_size']

        self.std = self.agent_config['gaussian_std']
        self.noise_clip = self.agent_config['noise_clip']
        self.reduce_rate = self.agent_config['noise_reduce_rate']
        self.gradient_steps = 0
        self.critic_steps = 0
        self.warm_up = self.agent_config['warm_up']

        self.actor_lr_main = self.agent_config['lr_actor']
        self.critic_lr_main = self.agent_config['lr_critic']

        self.actor_main, self.actor_target = Actor(self.act_space), Actor(self.act_space)
        self.actor_target.set_weights(self.actor_main.get_weights())
        self.actor_opt_main = Adam(self.actor_lr_main)
        self.actor_main.compile(optimizer=self.actor_opt_main)
        
        self.critic_main_1, self.critic_main_2 = Critic(), Critic()
        self.critic_target_1, self.critic_target_2 = Critic(), Critic()
        self.critic_target_1.set_weights(self.critic_main_1.get_weights())
        self.critic_target_2.set_weights(self.critic_main_2.get_weights())
        self.critic_opt_main_1 = Adam(self.critic_lr_main)
        self.critic_opt_main_2 = Adam(self.critic_lr_main)
        self.critic_main_1.compile(optimizer=self.critic_opt_main_1)
        self.critic_main_2.compile(optimizer=self.critic_opt_main_2)

    def action(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        # print('in action, obs: ', np.shape(np.array(obs)), obs)
        mu = self.actor_main(obs)
        # print('in action, mu: ', np.shape(np.array(mu)), mu)

        if self.gradient_steps > self.warm_up:
            std = tf.convert_to_tensor([self.std]*self.act_space, dtype=tf.float32)
            dist = tfp.distributions.Normal(loc=mu, scale=std)
            action = tf.squeeze(dist.sample())
            action = action.numpy()
            self.std = self.std * self.reduce_rate
        else:
            action = mu.numpy()[0]

        action = np.clip(np.clip(action, mu.numpy()[0] - self.noise_clip, mu.numpy()[0] + self.noise_clip), -1, 1)

        return action

    def update_target(self):
        actor_weights = []
        actor_targets = self.actor_target.weights
        
        for idx, weight in enumerate(self.actor_main.weights):
            actor_weights.append(weight * self.tau + actor_targets[idx] * (1 - self.tau))
        self.actor_target.set_weights(actor_weights)
        
        critic_weithgs_1 = []
        critic_targets_1 = self.critic_target_1.weights
        
        for idx, weight in enumerate(self.critic_main_1.weights):
            critic_weithgs_1.append(weight * self.tau + critic_targets_1[idx] * (1 - self.tau))
        self.critic_target_1.set_weights(critic_weithgs_1)

        critic_weithgs_2 = []
        critic_targets_2 = self.critic_target_2.weights
        
        for idx, weight in enumerate(self.critic_main_2.weights):
            critic_weithgs_2.append(weight * self.tau + critic_targets_2[idx] * (1 - self.tau))
        self.critic_target_2.set_weights(critic_weithgs_2)

    def update(self):
        if self.replay_buffer._len() < self.batch_size:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if not self.gradient_steps % self.update_freq == 0:  # only update every update_freq
            self.gradient_steps += 1
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        updated = True
        self.gradient_steps += 1
        self.critic_steps += 1

        actor_loss_val, criitic_loss1_val, ciritic_loss2_val = 0.0, 0.0, 0.0
        target_q_val, current_q_1_val, current_q_2_val = 0.0, 0.0, 0.0

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

        critic1_variable = self.critic_main_1.trainable_variables
        critic2_variable = self.critic_main_2.trainable_variables
        with tf.GradientTape() as tape_critic_1, tf.GradientTape() as tape_critic_2:
            tape_critic_1.watch(critic1_variable)
            tape_critic_2.watch(critic2_variable)
            target_action = self.actor_target(next_states)
            # print('target_action : {}'.format(target_action.numpy().shape), target_action)

            target_q_next_1 = tf.squeeze(self.critic_target_1(tf.concat([next_states,target_action], 1)), 1)
            target_q_next_2 = tf.squeeze(self.critic_target_2(tf.concat([next_states,target_action], 1)), 1)
            target_q_next = tf.math.minimum(target_q_next_1, target_q_next_2)
            # print('target_q_next_1 : {}'.format(target_q_next_1.numpy().shape), target_q_next_1)
            # print('target_q_next_2 : {}'.format(target_q_next_2.numpy().shape), target_q_next_2)
            # print('target_q_next : {}'.format(target_q_next.numpy().shape), target_q_next)

            target_q = tf.add(rewards, tf.multiply(self.gamma, tf.multiply(target_q_next, tf.subtract(1.0, tf.cast(dones, dtype=tf.float32)))))
            # print('target_q : {}'.format(target_q.numpy().shape), target_q)

            current_q_1 = tf.squeeze(self.critic_main_1(tf.concat([states,actions], 1)), 1)
            current_q_2 = tf.squeeze(self.critic_main_2(tf.concat([states,actions], 1)), 1)
            # print('current_q_1 : {}'.format(current_q_1.numpy().shape), current_q_1)
            # print('current_q_2 : {}'.format(current_q_2.numpy().shape), current_q_2)

            critic_loss_1 = tf.subtract(current_q_1, target_q)
            critic_loss_2 = tf.subtract(current_q_2, target_q)
            # print('critic_loss_1 : {}'.format(critic_loss_1.numpy().shape), critic_loss_1)
            # print('critic_loss_2 : {}'.format(critic_loss_2.numpy().shape), critic_loss_2)
            
            # (tf.abs(td_errors_1) + tf.abs(td_errors_2))/10 * is_weight
            td_errors = tf.cond(tf.convert_to_tensor(self.agent_config['use_PER'], dtype=tf.bool),\
                                lambda: tf.multiply(is_weight, tf.add(tf.multiply(0.5, tf.math.square(critic_loss_1)), tf.multiply(0.5, tf.math.square(critic_loss_2)))),\
                                lambda: tf.add(tf.multiply(0.5, tf.math.square(critic_loss_1)), tf.multiply(0.5, tf.math.square(critic_loss_2))))

            # td_errors = tf.multiply(is_weight, tf.add(tf.multiply(0.5, tf.math.square(critic_loss_1)), tf.multiply(0.5, tf.math.square(critic_loss_2))))
            # print('td_errors : {}'.format(td_errors.numpy().shape), td_errors)

            td_error = tf.math.reduce_mean(td_errors)
            # print('td_error : {}'.format(td_error.numpy().shape), td_error)
            
        grads_critic_1, _ = tf.clip_by_global_norm(tape_critic_1.gradient(td_error, critic1_variable), 0.5)
        grads_critic_2, _ = tf.clip_by_global_norm(tape_critic_2.gradient(td_error, critic2_variable), 0.5)

        self.critic_opt_main_1.apply_gradients(zip(grads_critic_1, critic1_variable))
        self.critic_opt_main_2.apply_gradients(zip(grads_critic_2, critic2_variable))

        criitic_loss1_val = tf.math.reduce_mean(critic_loss_1).numpy()
        ciritic_loss2_val = tf.math.reduce_mean(critic_loss_2).numpy()
        target_q_val  = tf.math.reduce_mean(target_q).numpy()
        current_q_1_val = tf.math.reduce_mean(current_q_1).numpy()
        current_q_2_val = tf.math.reduce_mean(current_q_2).numpy()

        if self.critic_steps % self.actor_update_freq == 0:

            actor_variable = self.actor_main.trainable_variables
            with tf.GradientTape() as tape_actor:
                tape_actor.watch(actor_variable)

                new_policy_actions = self.actor_main(states)
                # print('new_policy_actions : {}'.format(new_policy_actions.numpy().shape))
                actor_loss = -self.critic_main_1(tf.concat([states, new_policy_actions],1))
                # print('actor_loss : {}'.format(actor_loss.numpy().shape))
                actor_loss = tf.math.reduce_mean(actor_loss)
                # print('actor_loss : {}'.format(actor_loss.numpy().shape))

            grads_actor, _ = tf.clip_by_global_norm(tape_actor.gradient(actor_loss, actor_variable), 0.5)
            # grads_actor = tape_actor.gradient(actor_loss, self.actor_main.trainable_variables)        
            self.actor_opt_main.apply_gradients(zip(grads_actor, actor_variable))

            actor_loss_val = actor_loss.numpy()

        self.update_target()

        if self.agent_config['use_PER']:
            for i in range(self.batch_size):
                self.replay_buffer.update(idxs[i], td_errors[i].numpy())

        return updated, actor_loss_val, criitic_loss1_val, ciritic_loss2_val, target_q_val, current_q_1_val, current_q_2_val

    def save_xp(self, state, next_state, reward, action, done):
        # Store transition in the replay buffer.
        if self.agent_config['use_PER']:
            state_tf = tf.convert_to_tensor([state], dtype = tf.float32)
            action_tf = tf.convert_to_tensor([action], dtype = tf.float32)
            next_state_tf = tf.convert_to_tensor([next_state], dtype = tf.float32)
            target_action_tf = self.actor_target(next_state_tf)
            # print('state_tf: {}'.format(state_tf))
            # print('action_tf: {}'.format(action_tf))
            # print('next_state_tf: {}'.format(next_state_tf))
            # print('target_action_tf: {}'.format(target_action_tf))

            target_q_next_1 = tf.squeeze(self.critic_target_1(tf.concat([next_state_tf,target_action_tf], 1)), 1)
            target_q_next_2 = tf.squeeze(self.critic_target_2(tf.concat([next_state_tf,target_action_tf], 1)), 1)
            target_q_next = tf.math.minimum(target_q_next_1, target_q_next_2)
            # print('target_q_next_1: {}'.format(target_q_next_1))
            # print('target_q_next_2: {}'.format(target_q_next_2))
            # print('target_q_next: {}'.format(target_q_next))
            
            current_q_1 = tf.squeeze(self.critic_main_1(tf.concat([state_tf,action_tf], 1)), 1)
            current_q_2 = tf.squeeze(self.critic_main_2(tf.concat([state_tf,action_tf], 1)), 1)
            # print('current_q_1: {}'.format(current_q_1))
            # print('current_q_2: {}'.format(current_q_2))
            
            target_q = reward + self.gamma * target_q_next * (1.0 - tf.cast(done, dtype=tf.float32))
            # print('target_q: {}'.format(target_q))
            
            td_errors_1 = target_q - current_q_1
            td_errors_2 = target_q - current_q_2
            # print('td_errors_1: {}'.format(td_errors_1))
            # print('td_errors_2: {}'.format(td_errors_2))

            td_error = (0.5 * tf.math.square(td_errors_1) + 0.5 * tf.math.square(td_errors_2))
            # print('td_error: {}'.format(td_error))

            td_error = td_error.numpy()
            # print('td_error: {}'.format(td_error))

            self.replay_buffer.add(td_error[0], (state, next_state, reward, action, done))
        else:
            self.replay_buffer.add((state, next_state, reward, action, done))

    def load_models(self, path):
        print('Load Model Path : ', path)
        self.actor_main.load_weights(path, "_actor_main")
        self.actor_target.load_weights(path, "_actor_target")
        self.critic_main_1.load_weights(path, "_critic_main_1")
        self.critic_main_2.load_weights(path, "_critic_main_2")
        self.critic_target_1.load_weights(path, "_critic_target_1")
        self.critic_target_2.load_weights(path, "_critic_target_2")

    def save_models(self, path, score):
        save_path = str(path) + "score_" + str(score) + "_model"
        print('Save Model Path : ', save_path)
        self.actor_main.save_weights(save_path, "_actor")
        self.actor_target.save_weights(save_path, "_actor_target")
        self.critic_main_1.save_weights(save_path, "_critic_main_1")
        self.critic_main_2.save_weights(save_path, "_critic_main_2")
        self.critic_target_1.save_weights(save_path, "_critic_target_1")
        self.critic_target_2.save_weights(save_path, "_critic_target_2")
