import os

import numpy as np
import cvxpy as cp

from algorithms import vi_policy


if __name__ == '__main__':
    P = np.load('../data/P.npy')
    traj = np.load('../data/traj.npy')
    goal = np.load('../data/goal.npy')

    num_states = P.shape[0]
    gridsize = int(np.sqrt(num_states))
    num_actions = P.shape[-1]
    num_demo = traj.shape[0]
    bkps = [0] + list(np.where(np.diff(goal))[0] + 1) + [num_demo]

    gamma = 0.9  # discount factor
    rmax = 100  # reward function bound
    lbd = 0.5  # scalarization weight

    output_root = '../outputs'
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    r_glbs = []
    a_est = []
    solve_times = []
    for bkp_idx in range(1, len(bkps)):
        subtraj = traj[bkps[bkp_idx - 1]: bkps[bkp_idx]]
        pi = np.zeros((num_states, num_actions))
        substates = np.unique(np.hstack([subtraj[:, 0], subtraj[:, -1]]))
        num_substates = len(substates)

        for s, a, ns in subtraj:
            pi[s, a] += 1
        pi = pi[substates]
        pi /= np.sum(pi, axis=-1, keepdims=True)
        
        tilP = P[:, substates, :][substates, :, :]
        for a in range(num_actions):
            for s in range(num_substates):
                tilP[s, :, a] /= np.sum(tilP[s, :, a], axis=-1)
        Pastr = []
        lPa = []
        for s_idx in range(num_substates):
            if len(np.unique(pi[s_idx, :])) == 1:
                astr = np.random.choice(num_actions)
            else:
                astr = np.argmax(pi[s_idx, :])
            a = np.delete(np.arange(num_actions), astr)
            Pastr.append(tilP[s_idx, :, astr])
            lPa.append(tilP[s_idx, :, a])
        Pastr = np.array(Pastr)
        lPa = np.array(lPa).transpose(1, 0, 2)

        m = num_substates  # number of (sub)states
        r = cp.Variable(m)
        s = cp.Variable(m)

        constraints = []
        H = np.linalg.inv(np.identity(m) - gamma * Pastr)
        D = np.array([[Pastr[i] - Pa[i] for Pa in lPa] for i in range(m)])
        for i in range(m):
            constraints.append(D[i] @ H @ r + s[i] >= 0)
        for Pa in lPa:
            constraints.append((Pastr - Pa) @ H @ r >= 0)
        constraints.append(rmax >= r)
        constraints.append(r >= 0)

        obj = cp.Minimize(cp.sum(s) + lbd * cp.norm(r, 1))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        print(f'problem value: {prob.value:.2f}; time: {prob.solver_stats.solve_time}')
        solve_times.append(prob.solver_stats.solve_time)

        r_glb = np.zeros(num_states)
        for s_idx, s in enumerate(substates):
            r_glb[s] = r.value[s_idx]
        r_glbs.append(r_glb)
        pi_hat = vi_policy(num_states=num_substates, num_actions=num_actions,
                P=tilP, reward=r.value, discount=gamma, stochastic=False, threshold=1e-2)
        for s, a, ns in subtraj:
            a_est.append(np.argmax(pi_hat[np.where(substates == s)[0]]))
    print(f'fraction of action matching: {np.mean(traj[:, 1] == a_est):.2f}')
    np.save(os.path.join(output_root, f'r_glbs.npy'), np.array(r_glbs))
    np.save(os.path.join(output_root, f'a_est.npy'), np.array(a_est))
    np.save(os.path.join(output_root, f'solve_times.npy'), np.array(solve_times))
