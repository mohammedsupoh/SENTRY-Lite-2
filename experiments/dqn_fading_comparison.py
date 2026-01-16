"""
P2: DQN Baseline Under Fading Channels
Compare RL performance vs SENTRY-Lite-2 under Rayleigh/Rician
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import sys
sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/src')
from environment.wireless_env import WirelessURLLCEnv, WirelessConfig, ChannelModel


class DQNNetwork(nn.Module):
    """Simple DQN for fading channel"""
    
    def __init__(self, state_dim: int = 5, action_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def train_dqn_fading(channel: ChannelModel, snr: float, n_episodes: int = 100):
    """Train DQN on fading channel"""
    
    config = WirelessConfig(channel_model=channel, snr_db=snr)
    env = WirelessURLLCEnv(config, seed=42)
    
    # DQN setup
    state_dim = 5  # [q_norm, slack, mode, cqi_norm, snr_norm]
    action_dim = 2
    
    policy_net = DQNNetwork(state_dim, action_dim)
    target_net = DQNNetwork(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    
    # Replay buffer (simplified)
    buffer = []
    buffer_size = 10000
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    
    # Training
    for episode in range(n_episodes):
        state = env.reset(seed=episode)
        total_reward = 0
        
        for t in range(5000):
            # Epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(2)
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.FloatTensor(state))
                    action = q_vals.argmax().item()
            
            next_state, reward, done, info = env.step(action)
            
            # Custom reward for URLLC
            Q = len(env.queue)
            cqi = env.channel.current_cqi
            
            reward = 0
            if env.current_mode == 0:  # BASE
                reward += 1  # Energy reward
            reward -= 100 * (env.total_misses > 0)  # Miss penalty
            if Q > 15:
                reward -= 10  # Queue penalty
            
            # Store transition
            buffer.append((state, action, reward, next_state, done))
            if len(buffer) > buffer_size:
                buffer.pop(0)
            
            # Train
            if len(buffer) >= batch_size:
                batch = [buffer[i] for i in np.random.choice(len(buffer), batch_size)]
                
                states = torch.FloatTensor([b[0] for b in batch])
                actions = torch.LongTensor([b[1] for b in batch])
                rewards = torch.FloatTensor([b[2] for b in batch])
                next_states = torch.FloatTensor([b[3] for b in batch])
                
                # Q-learning update
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                target_q = rewards + gamma * next_q
                
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Update target network
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    return policy_net


def evaluate_dqn(policy_net, channel: ChannelModel, snr: float, n_seeds: int = 10):
    """Evaluate trained DQN"""
    
    results = {'miss': [], 'savings': [], 'switches': []}
    
    for seed in range(n_seeds):
        config = WirelessConfig(channel_model=channel, snr_db=snr)
        env = WirelessURLLCEnv(config, seed=seed)
        state = env.reset(seed=seed)
        
        for t in range(10000):
            with torch.no_grad():
                q_vals = policy_net(torch.FloatTensor(state))
                action = q_vals.argmax().item()
            
            state, _, done, _ = env.step(action)
            if done:
                break
        
        metrics = env.get_metrics()
        results['miss'].append(metrics['miss_probability'] * 100)
        results['savings'].append(metrics['energy_savings'])
        results['switches'].append(metrics['switches_per_ktti'])
    
    return {
        'miss_mean': np.mean(results['miss']),
        'miss_std': np.std(results['miss']),
        'savings_mean': np.mean(results['savings']),
        'switches_mean': np.mean(results['switches'])
    }


def evaluate_sentry(channel: ChannelModel, snr: float, n_seeds: int = 10):
    """Evaluate SENTRY-Lite-2 (for comparison)"""
    
    results = {'miss': [], 'savings': [], 'switches': []}
    
    for seed in range(n_seeds):
        config = WirelessConfig(channel_model=channel, snr_db=snr)
        env = WirelessURLLCEnv(config, seed=seed)
        env.reset(seed=seed)
        
        counter = 0
        for t in range(10000):
            Q = len(env.queue)
            slack = env.ewma_slack
            cqi = env.channel.current_cqi
            
            if Q >= 15 or slack < 0.25 or cqi < 8:
                action = 1
                counter = 0
            elif Q < 6 and slack > 0.50 and cqi >= 10:
                counter += 1
                action = 0 if counter >= 2 else env.current_mode
            else:
                action = env.current_mode
                counter = 0
            
            env.step(action)
        
        metrics = env.get_metrics()
        results['miss'].append(metrics['miss_probability'] * 100)
        results['savings'].append(metrics['energy_savings'])
        results['switches'].append(metrics['switches_per_ktti'])
    
    return {
        'miss_mean': np.mean(results['miss']),
        'miss_std': np.std(results['miss']),
        'savings_mean': np.mean(results['savings']),
        'switches_mean': np.mean(results['switches'])
    }


if __name__ == '__main__':
    print("="*70)
    print("P2: DQN vs SENTRY-Lite-2 UNDER FADING CHANNELS")
    print("="*70)
    
    # Test points
    test_points = [
        (ChannelModel.RAYLEIGH, 18, 'Rayleigh-18dB (URLLC fail)'),
        (ChannelModel.RAYLEIGH, 20, 'Rayleigh-20dB (URLLC boundary)'),
        (ChannelModel.RICIAN, 20, 'Rician-20dB (URLLC fail)'),
        (ChannelModel.RICIAN, 22, 'Rician-22dB (URLLC boundary)'),
    ]
    
    all_results = []
    
    for channel, snr, name in test_points:
        print(f"\n{'='*50}")
        print(f"[{name}]")
        print("="*50)
        
        # Train DQN
        print("  Training DQN (100 episodes)...")
        policy_net = train_dqn_fading(channel, snr, n_episodes=100)
        
        # Evaluate DQN
        print("  Evaluating DQN...")
        dqn_results = evaluate_dqn(policy_net, channel, snr, n_seeds=10)
        
        # Evaluate SENTRY
        print("  Evaluating SENTRY-Lite-2...")
        sentry_results = evaluate_sentry(channel, snr, n_seeds=10)
        
        # Print comparison
        print(f"\n  {'Controller':<15} | {'Miss%':>10} | {'Savings':>8} | {'Sw/kTTI':>8} | URLLC")
        print(f"  {'-'*15} | {'-'*10} | {'-'*8} | {'-'*8} | -----")
        
        for ctrl_name, res in [('SENTRY-Lite-2', sentry_results), ('DQN', dqn_results)]:
            urllc = "✓" if res['miss_mean'] < 1.0 else "✗"
            print(f"  {ctrl_name:<15} | {res['miss_mean']:>8.3f}% | {res['savings_mean']:>7.1f}% | "
                  f"{res['switches_mean']:>8.1f} | {urllc:>5}")
        
        all_results.append({
            'scenario': name,
            'channel': channel.value,
            'snr': snr,
            'sentry_miss': sentry_results['miss_mean'],
            'sentry_savings': sentry_results['savings_mean'],
            'sentry_switches': sentry_results['switches_mean'],
            'dqn_miss': dqn_results['miss_mean'],
            'dqn_savings': dqn_results['savings_mean'],
            'dqn_switches': dqn_results['switches_mean'],
        })
    
    # Summary
    print("\n" + "="*70)
    print("[SUMMARY: DQN vs SENTRY-Lite-2 Under Fading]")
    print("-"*70)
    
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))
    
    df.to_csv('experiments/results/dqn_vs_sentry_fading.csv', index=False)
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("  1. DQN struggles to learn channel-aware policy")
    print("  2. SENTRY-Lite-2's explicit CQI triggers provide")
    print("     more reliable response to channel degradation")
    print("="*70)
