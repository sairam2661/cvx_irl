import os

import numpy as np

from gridworld import GridWorld
from algorithms import vi_policy


if __name__ == '__main__':
    gridsize = 16
    wind = 0.1
    np.random.seed(10015)

    output_root = '../data'
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    envr = GridWorld(size=gridsize, wind=wind)
    np.save(os.path.join(output_root, 'P.npy'), envr.P)
    # P = np.load('../data/P.npy')
    # envr = GridWorld(size=gridsize, wind=wind, P=P)

    r = np.zeros(envr.num_states)
    r[envr.goals] += 100
    pi = vi_policy(num_states=envr.num_states, num_actions=envr.num_actions,
                   P=envr.P, reward=r, discount=envr.gamma, stochastic=False, threshold=1e-2)
    np.save(os.path.join(output_root, 'r.npy'), r)
    np.save(os.path.join(output_root, 'pi.npy'), pi)
