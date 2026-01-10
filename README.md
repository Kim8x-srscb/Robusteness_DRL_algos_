# DRL Algorithm Robustness Under Stochasticity

## Overview
This project evaluates the robustness of Deep Reinforcement Learning (DRL) algorithms to environmental stochasticity and faults. We test how well agents trained on clean environments generalize to noisy/faulty conditions, and measure their resilience across different failure modes.

## Project Structure
```
├── LunarLander-v2/
│   ├── DQN/
│   │   ├── DQN_2exp_final.ipynb      # Main DQN training & evaluation
│   │   ├── models/
│   │   │   ├── exp1/                 # 30 trained agents (cross-validation)
│   │   │   └── exp2/                 # 5 agents (zero-shot evaluation)
│   │   └── results/
│   │       ├── exp1/                 # Cross-validation results + plots
│   │       └── exp2/                 # Zero-shot results + GIFs
│   └── PPO/
│       ├── PPO_2exp_final.ipynb      # Main PPO training & evaluation
│       ├── models/
│       │   ├── exp1_ppo/             # 30 trained agents (cross-validation)
│       │   └── exp2_ppo/             # 5 agents (zero-shot evaluation)
│       └── results/
│           ├── exp1_ppo/             # Cross-validation results + plots
│           └── exp2_ppo/             # Zero-shot results + GIFs
```

## Algorithms Implemented

### DQN (Deep Q-Network)
Value-based off-policy algorithm using experience replay and target networks.

### PPO (Proximal Policy Optimization)
Policy gradient on-policy algorithm with clipped objective for stable training.

## Experiments

Both algorithms follow the same experimental protocol:

- **Experiment 1:** Train 30 agents (6 failure probabilities × 5 seeds), evaluate cross-all-probabilities
- **Experiment 2:** Train 5 agents (p=0.0 only), evaluate zero-shot on all probabilities

## Failure Modes (4 stochastic families)
1. **Sensor failures:** Altimeter drift, gyroscope malfunction, velocity errors
2. **Actuator failures:** Engine degradation, thruster asymmetry, control lag
3. **Dynamics failures:** Fuel leaks, gravity variation, wind turbulence
4. **Catastrophic failures:** Engine cutoff, landing gear damage

## Training Configuration

### DQN
```python
TRAINING_TIMESTEPS = 700_000
FAILURE_PROBS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
N_EVAL_EPISODES = 100

DQN_CONFIG = {
    'learning_rate': 1e-3,
    'buffer_size': 100_000,
    'batch_size': 64,
    'gamma': 0.99,
    'target_update_interval': 10_000,
    'exploration_fraction': 0.2,
    'exploration_final_eps': 0.05
}
```

### PPO
```python
TRAINING_TIMESTEPS = 700_000
FAILURE_PROBS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
N_EVAL_EPISODES = 100

PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2
}
```

## Metrics Tracked
- **Success rate:** Episodes with return ≥ 200
- **Crash rate:** Episodes with return < -100
- **Time-to-failure:** Steps until first stochastic event (Exp2 only)
- **Pre/post-failure returns:** Decomposed episode performance (Exp2 only)

## Output Files
Each algorithm produces:
- `results/exp*/exp*_results.csv` – Evaluation results
- `results/exp*/plots/` – Heatmaps, robustness curves, performance plots
- `results/exp2*/gifs/` – Episode visualizations (2× speed)

## Running Experiments

### DQN
Open `LunarLander-v2/DQN/DQN_2exp_final.ipynb`:
```python
# Load and evaluate existing results:
jupyter_evaluate_exp1()  # Cross-evaluation
jupyter_evaluate_exp2()  # Zero-shot evaluation

# Or full pipeline (trains + evaluates):
run_complete_pipeline()
```

### PPO
Open `LunarLander-v2/PPO/PPO_2exp_final.ipynb`:
```python
# Load and evaluate existing results:
jupyter_evaluate_exp1()  # Cross-evaluation
jupyter_evaluate_exp2()  # Zero-shot evaluation

# Or full pipeline (trains + evaluates):
run_complete_pipeline()
```

## Key Findings

### DQN
- Models trained with zero stochasticity maintain robustness up to ~25% failure probability
- On-distribution training improves success rate by ~5pp compared to zero-shot
- Catastrophic failures have largest performance impact; sensor failures are most resilient

### PPO
- (Results available after running experiments)

---

**Next steps:** Implement A2C for additional comparison
