import os
import sys

import pygame

import ppo
import numpy as np
from matplotlib import pyplot as plt

# os.environ['SDL_VIDEODRIVER']='dummy' # if you want to use the codes in
# online notebook like Kaggle/Colab, should as this line to omit the pygame video render

ARG_TRAIN = 0
ARG_AI = 1


def plot_info(ppo_agent):
    print(">>> Loading plot info ...")

    print(">>> TRAIN GAME END >>>")
    # print(np.array(self.returns))
    plt.figure()
    plt.plot(np.arange(len(ppo_agent.returns)) * 20, np.array(ppo_agent.returns))
    plt.plot(np.arange(len(ppo_agent.returns)) * 20, np.array(ppo_agent.returns), 's')
    plt.xlabel('Episodes')
    plt.ylabel('returns')
    plt.show()

    plt.plot(np.arange(0, len(ppo_agent.policy_loss_h), 1), ppo_agent.policy_loss_h[::1])
    plt.xlabel('Episode')
    plt.ylabel('policy loss')
    plt.show()

    plt.plot(np.arange(0, len(ppo_agent.value_loss_h), 1), ppo_agent.value_loss_h[::1])
    plt.xlabel('Episode')
    plt.ylabel('value loss')
    plt.show()

    plt.plot(np.arange(0, len(ppo_agent.scores_history), 1), ppo_agent.scores_history[::1])
    plt.xlabel('Episode')
    plt.ylabel('scores')
    plt.show()

    print("<<< TRAIN DONE <<<")


if __name__ == '__main__':
    actor = ppo.Actor()
    actor.compile()
    actor.build(input_shape=(None, 1, ppo.STATE_DIM))

    critic = ppo.Critic()
    critic.compile()
    critic.build(input_shape=(None, 1, ppo.STATE_DIM))

    agent = None

    args = int(input("Train or AI? - [enter 0 or 1]:"))
    if args == ARG_TRAIN:
        agent = ppo.PPO(actor=actor, critic=critic)
        agent.execute_train()
    elif args == ARG_AI:
        # load NN weights from .hdf5 in "model dir"
        print("Loading model weights...")
        actor.load_weights('./model/ppo_actor.hdf5')
        critic.load_weights('./model/ppo_critic.hdf5')
        agent = ppo.PPO(actor=actor, critic=critic)
        agent.execute_ai()
    else:
        print("Invalid Operation! Try again.")

    plot_info(agent)
    pygame.quit()
    print('-' * 25)
    print("<<< ALL DONE <<<")
    sys.exit(0)
