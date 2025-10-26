# Convex Inverse Reinforcement Learning

Implementation of noise-robust inverse reinforcement learning using convex optimization.

## Overview

This project extends the Convex Inverse Reinforcement Learning (CIRL) framework to handle noisy and suboptimal expert demonstrations - a common challenge in real-world applications. Unlike traditional non-convex IRL methods that can converge to local optima, our convex formulation guarantees global optimality while achieving superior computational efficiency.

## Key Features

- **Noise-Robust Formulation**: Extends standard CIRL with slack variables and regularization to handle noisy demonstrations
- **Global Optimality**: Convex formulation ensures convergence to global optimum with KKT condition verification
- **Computational Efficiency**: 5-10x faster than MaxEnt IRL while maintaining better performance
- **Multiple Environments**: Tested on GridWorld and Greedy Snake environments with varying noise levels (5-30%)

## Approach

Our method introduces three key modifications to standard CIRL:

1. **Slack Variables**: Allow controlled constraint violations for noisy demonstrations
```
   (P_{a*} - P_a)(I - γP_{a*})^{-1} r ≥ -ε_a
```

2. **Penalty Terms**: Discourage unnecessary violations while maintaining convexity
```
   minimize: J(r) + λ₁||r||₁ + λ_noise Σε_a + λ_L2||r||₂²
```

3. **L2 Regularization**: Promotes smoother reward functions less sensitive to individual noisy demonstrations

## Results

### GridWorld Performance

| Noise Level | Original CIRL | Noise-Robust CIRL | MaxEnt IRL |
|-------------|---------------|-------------------|------------|
| 5%          | 92% / 0.80    | 86% / 0.81       | 33% / 0.48 |
| 15%         | 45% / 0.39    | 69% / -0.26      | 26% / 0.47 |
| 30%         | 43% / 0.23    | 61% / -0.20      | 25% / 0.43 |

*Policy Match (%) / Reward Similarity*

**Key Findings:**
- **Low noise (5-10%)**: Original CIRL performs best
- **High noise (15-30%)**: Noise-Robust CIRL maintains 61-69% policy match vs 43-54% for original CIRL
- **Runtime**: Convex methods ~1-3 seconds vs ~14 seconds for MaxEnt IRL
- **Crossover point**: ~15% noise where our robust formulation begins significantly outperforming standard CIRL

## Installation
```bash
pip install numpy==1.26.4 cvxpy==1.6.0 matplotlib==3.10.0 seaborn==0.13.2
```

For video generation in Example 2:
```bash
# Install ffmpeg (platform-dependent)
```

## Technical Details

### Problem Formulation

**Primal:**
```
minimize    J(r) + λ||r||₁ + λ_noise Σε_a + λ_L2||r||₂²
subject to  (P_{a*} - P_a)(I - γP_{a*})^{-1}r ≥ -ε_a  ∀a ∈ A\{a*}
            -r_max ≤ r ≤ r_max
            ε_a ≥ 0  ∀a ∈ A\{a*}
```

Where:
- `J(r)`: Margin maximization between expert policy and alternatives
- `P_{a*}, P_a`: Transition matrices for expert and alternative actions
- `γ`: Discount factor
- `ε_a`: Slack variables for noise tolerance

### Convexity Preservation

The formulation maintains convexity through:
1. Linear constraints with slack variables
2. Convex regularization terms (L1 and L2 norms)
3. Convex objective function

This allows efficient solving via CVXPY with guaranteed global optimality.

## Comparison with Non-Convex Methods

| Metric | Noise-Robust CIRL | MaxEnt IRL |
|--------|-------------------|------------|
| Optimality | Global (verified via KKT) | Local optima possible |
| Runtime | ~1-3 seconds | ~14 seconds |
| Noise Robustness | Strong (61% at 30% noise) | Weak (25% at 30% noise) |
| Convergence | Guaranteed | Gradient-dependent |

## Implementation Notes

- Built using CVXPY for domain-specific convex optimization
- KKT conditions verified for all solutions
- Handles both analytical policies and trajectory-based demonstrations
- Automatic hyperparameter selection via Pareto analysis

## Future Work

- Extension to multi-agent systems
- Handling partial observability
- Theoretical guarantees under adversarial demonstrations
- Open-source benchmarking tools

## References

Based on the (convex IRL)[https://arxiv.org/pdf/2501.15957] formulation by Ng and Russell, with extensions for noise robustness.
