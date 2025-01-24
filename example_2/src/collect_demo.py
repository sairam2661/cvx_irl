import os

import numpy as np

from gridworld import GridWorld
from algorithms import vi_policy


if __name__ == '__main__':
    gridsize = 48
    wind = 0.1
    num_demo = 2000
    np.random.seed(0)

    output_root = '../data'
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    envr = GridWorld(size=gridsize, wind=wind)
    np.save(os.path.join(output_root, 'P.npy'), envr.P)

    goal = []
    traj = []
    s, g = envr.reset()
    R = np.zeros(envr.num_states)
    R[g] = 100
    pi = vi_policy(num_states=envr.num_states, num_actions=envr.num_actions,
                P=envr.P, reward=R, discount=envr.gamma, stochastic=False, threshold=1e-1)

    t = 0
    while True:
        a = np.argmax(pi[s])
        ns, ng, r = envr.step(a)
        traj.append([s, a, ns])
        goal.append(g)
        s = ns
        g = ng
        t += 1
        if r == 1:
            if t > num_demo:
                break
            else:
                R = np.zeros(envr.num_states)
                R[g] = 100
                pi = vi_policy(num_states=envr.num_states, num_actions=envr.num_actions,
                            P=envr.P, reward=R, discount=envr.gamma, stochastic=False, threshold=1e-1)
    np.save(os.path.join(output_root, f'traj.npy'), np.array(traj))
    np.save(os.path.join(output_root, f'goal.npy'), np.array(goal))
