import numpy as np


def value_iteration(reward, P, num_states, num_actions, discount, threshold=1e-2):
    """
    calculate the optimal state value function of given enviroment.

    :param reward: reward vector. nparray. (states, )
    :param P: transition probability p(st | s, a). nparray. (states, states, actions).
    :param discount: discount rate gamma. float. Default: 0.99
    :param num_states: number of states. int.
    :param num_actions: number of actions. int.
    :param threshold: stop when difference smaller than threshold. float.
    :return: optimal state value function. nparray. (states)
    """

    v = np.zeros(num_states)

    while True:
        delta = 0

        for s in range(num_states):
            max_v = float("-inf")
            for a in range(num_actions):
                tp = P[s, :, a]
                max_v = max(max_v, np.dot(tp, (reward + discount * v)))

            diff = abs(v[s] - max_v)
            delta = max(delta, diff)

            v[s] = max_v

        if delta < threshold:
            break

    return v


def vi_policy(num_states, num_actions, P, reward, discount, stochastic=True, threshold=1e-2):
    """
    Find the optimal policy.

    num_states: Number of states. int.
    num_actions: Number of actions. int.
    P: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    v = value_iteration(reward, P, num_states, num_actions, discount, threshold)

    policy = np.zeros((num_states, num_actions))
    if stochastic:
        for s in range(num_states):
            for a in range(num_actions):
                p = P[s, :, a]
                policy[s, a] = p.dot(reward + discount*v)
        policy -= policy.max(axis=1).reshape((num_states, 1))  # For numerical stability.
        policy = np.exp(policy)/np.exp(policy).sum(axis=1).reshape((num_states, 1))

    else:
        def _policy(s):
            return max(range(num_actions),
                       key=lambda a: sum(P[s, k, a] *
                                         (reward[k] + discount * v[k])
                                         for k in range(num_states)))
        for s in range(num_states):
            policy[s, _policy(s)] = 1
    return policy
