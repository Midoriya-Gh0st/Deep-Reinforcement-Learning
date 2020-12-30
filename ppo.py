import random

import pygame
from matplotlib import pyplot as plt

import game
from graphical import Game

import gym
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from collections import namedtuple

assert tf.__version__.startswith('2.')

# gamma = 0.98
epsilon = 0.2  # PPO hyperparams: (1-0.2, 1+0.2)
batch_size = 20  # batch size

TRAIN_EPISODES = 100  # number of overall episodes for training (game times)
TEST_EPISODES = 100  # number of overall episodes for testing
MAX_STEPS = 25  # maximum time step in one episode (game)
GAMMA = 0.98  # reward discount
LR_A = 1e-5  # learning rate for actor
LR_C = 3e-5  # learning rate for critic

CMD_TRAIN = 0
CMD_TEST = 1

STATE_DIM = 8 * 10 * 5
ACTION_DIM = 142  # 10*(8-1) + 8*(10-1)

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(keras.Model):
    def __init__(self):
        super(Actor, self).__init__()
        # actor (policy) NN
        # output the probability distribution -> pi(a|s)
        input_layer = layers.Input([1, STATE_DIM])
        fc1 = layers.Dense(256, kernel_initializer='he_normal', activation=tf.nn.relu)(input_layer)
        fc2 = layers.Dense(ACTION_DIM, kernel_initializer='he_normal', activation=tf.nn.softmax)(fc1)

        self.model = keras.models.Model(inputs=input_layer, outputs=fc2)
        self.optimizer = tf.optimizers.Adam(LR_A)

    def call(self, inputs, **kwargs):
        x = self.model(inputs)
        return x


class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        # Critic NN -> output the v(s)
        input_layer = layers.Input([1, STATE_DIM])
        fc1 = layers.Dense(256, kernel_initializer='he_normal', activation=tf.nn.relu)(input_layer)
        fc2 = layers.Dense(1, kernel_initializer='he_normal')(fc1)

        self.model = keras.models.Model(inputs=input_layer, outputs=fc2)
        self.optimizer = tf.optimizers.Adam(LR_C)

    def call(self, inputs, **kwargs):
        x = self.model(inputs)
        return x


class PPO:
    # PPO算法主体
    def __init__(self, actor, critic):
        # super().__init__()
        self.actor = actor
        self.critic = critic
        self.buffer = []  # Data Buffer
        self.actor_optimizer = optimizers.Adam(LR_A)
        self.critic_optimizer = optimizers.Adam(LR_C)

        self.env = Game(self.ai_callback, self.transition_callback, self.end_of_game_callback, speed=1000.0, seed=None)
        self.cmd = CMD_TRAIN  # train or AI (test)
        self.action = None  # random.randint(0, ACTION_DIM-1)
        self.action_prob = None
        self.move = None
        self.total = None
        self.done = None
        self.episode = 0
        self.returns = None
        self.policy_loss_h = []
        self.value_loss_h = []
        self.scores_history = []

    def select_action(self, s):
        # input the 'state', get policy: [4]
        s = tf.constant(to_num(s), dtype=tf.float32)
        # s: [4] => [1,4]
        s = tf.expand_dims(s, axis=0)
        # get policy distribution
        prob = self.actor(s)
        # get one action, shape: [1]
        a = tf.random.categorical(tf.math.log(prob), 1)[0]
        a = int(a)  # Tensor->Num
        return a, float(prob[0][a])  # return action & prob

    def get_value(self, s):
        s = tf.constant(to_num(s), dtype=tf.float32)
        s = tf.expand_dims(s, axis=0)
        v = self.critic(s)[0]
        return float(v)  # return v(s)

    def store_transition(self, transition):
        # store the samples (fullfil the buffer)
        self.buffer.append(transition)

    def optimize(self):
        state = tf.constant([to_num(t.state) for t in self.buffer], dtype=tf.float32)
        action = tf.constant([t.action for t in self.buffer], dtype=tf.int32)
        action = tf.reshape(action, [-1, 1])
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = tf.constant([t.a_log_prob for t in self.buffer], dtype=tf.float32)
        old_action_log_prob = tf.reshape(old_action_log_prob, [-1, 1])

        R = 0
        Rs = []
        for r in reward[::-1]:
            R = r + GAMMA * R
            Rs.insert(0, R)
        Rs = tf.constant(Rs, dtype=tf.float32)
        # 10 iterations for data in the buffer
        for _ in range(round(10 * len(self.buffer) / batch_size)):
            # sample BATCH_SIZE samples from the buffer randomly
            index = np.random.choice(np.arange(len(self.buffer)), batch_size, replace=False)
            """IMPORTANT"""
            # construct the GradientTape to track the observation
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                # get R(st)，[b,1]
                v_target = tf.expand_dims(tf.gather(Rs, index, axis=0), axis=1)
                # get v(s) predict
                v = self.critic(tf.gather(state, index, axis=0))
                delta = v_target - v  # get Advantage
                advantage = tf.stop_gradient(delta)
                a = tf.gather(action, index, axis=0)  # get AT of batch
                # action distributio of batch = pi(a|st)
                pi = self.actor(tf.gather(state, index, axis=0))
                indices = tf.expand_dims(tf.range(a.shape[0]), axis=1)
                indices = tf.concat([indices, a], axis=1)
                pi_a = tf.gather_nd(pi, indices)  # action prob = pi(at|st), [b]
                pi_a = tf.expand_dims(pi_a, axis=1)  # [b]=> [b,1]
                # ratio
                ratio = (pi_a / tf.gather(old_action_log_prob, index, axis=0))
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage
                # PPO loss
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                value_loss = losses.MSE(v_target, v)  # NB. MSE (mean(squar()))
                self.policy_loss_h.append(policy_loss)  # history record for plot
                self.value_loss_h.append(value_loss)
            # optimize the policy NN
            grads = tape1.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
            # optimize the
            grads = tape2.gradient(value_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        self.buffer = []  # buffer clear

    def ai_callback(self, board, score,
                    moves_left):  # will auto-ly execute env.step(move) -> get by transition_callback()
        self.action, self.action_prob = self.select_action(board)
        self.move = get_move(action=self.action, board=board)

        return self.move

    def transition_callback(self, board, move, score_delta, next_board, moves_left):
        # next_state, step -> return(obs) -> executed_obs
        if moves_left == 0:
            self.done = 1
        else:
            self.done = 0
        next_state, reward = next_board, score_delta
        # print("1 - ppo.py - transition - done:", self.done)

        # trans = Transition(state, action, action_prob, reward, next_state)
        trans = Transition(board, self.action, self.action_prob, reward, next_state)

        self.store_transition(trans)
        # state = next_state  # refresh the state - at game_logic
        self.total += reward  # accumulated reward
        # This can be used to monitor outcomes of moves

    def end_of_game_callback(self, boards, scores, moves, final_score):  # executed after one episode (i.e. 25 steps)
        self.done = 1
        self.episode += 1
        if self.done:
            if len(self.buffer) >= batch_size:
                self.optimize()  # train the NN
            # break (end the game)
            # self.returns_each.append(self.total)
            if self.episode % 20 == 1:  # check ave_returns each 20 episodes
                self.returns.append(self.total / 20)
                self.total = 0
                print("Episode:", self.episode, "-> Return:", self.returns[-1])

        # print("1 - ppo.py - end of game - done:", self.done)

        self.scores_history.append(final_score)

        if self.cmd == CMD_TRAIN:
            if self.episode >= TRAIN_EPISODES - 1:
                # print("the episode:", self.episode)
                # input("end of game?")
                save = int(input("Save network model? - [1/0]"))
                if save == 1:
                    self.save_ckpt()
                # self.actor.summary()
                # self.critic.summary()
                return False

        if self.cmd == CMD_TEST:
            if self.episode >= TEST_EPISODES - 1:
                # self.actor.summary()
                # self.critic.summary()
                return False

        return True  # True = play another, False = Done

    def execute_train(self):
        self.returns = []  # calculate the total returns
        self.total = 0  # total returns during a period
        self.env.run()
        pygame.quit()

    def execute_ai(self):
        self.returns = []
        self.total = 0
        self.cmd = CMD_TEST
        self.env.run()
        pygame.quit()

    def save_ckpt(self):
        print(">>> Saving model checkpoint...")
        if not os.path.exists('model'):
            os.makedirs('model')
        self.actor.save_weights('./model/ppo_actor.hdf5')
        self.critic.save_weights('./model/ppo_critic.hdf5')

    # transform the board from string to num


# 10-row, 8-col
def to_num(scn):
    screen = scn.split('\n')
    board = np.zeros([10, 8])
    len_row = len(screen)
    len_col = len(screen[0])

    for i in range(len_row):  # row
        for j in range(len_col):  # col
            if screen[i][j] == 'a':
                board[i][j] = 0
            if screen[i][j] == 'b':
                board[i][j] = 1
            if screen[i][j] == 'c':
                board[i][j] = 2
            if screen[i][j] == 'd':
                board[i][j] = 3
            if screen[i][j] == '#':
                board[i][j] = 4

    board_0 = np.copy(board)
    board_1 = np.copy(board)
    board_2 = np.copy(board)
    board_3 = np.copy(board)
    board_4 = np.copy(board)

    board_0[board_0 != 0] = -1
    board_1[board_1 != 1] = -1
    board_2[board_2 != 2] = -1
    board_3[board_3 != 3] = -1
    board_4[board_4 != 4] = -1

    boards = np.array([board_0, board_1, board_2, board_3, board_4])
    new_boards = np.resize(boards, (8 * 10 * 5))
    return new_boards


# map the actor to an unified move
# more vividly pls google keyword "How King Tech Reinforcement Learning"
def get_move(action, board):
    y, x, d = 0, 0, 0
    if 0 <= action <= 71:
        d = 1
        x = int(action / (game.HEIGHT - 1))  # /9:
        y = int(action % (game.HEIGHT - 1))  # %9:
        # (min=(0, 0) -> max=(7, 0)) ↓ (7, 8)

    elif 72 <= action <= 141:
        i = action - 72
        d = 0
        x = int(i / game.HEIGHT)
        y = int(i % game.HEIGHT)
        # (min=(0, 0) ↓ max=(0, 9)) -> (6, 9)

    move = (x, y, d)

    return move
