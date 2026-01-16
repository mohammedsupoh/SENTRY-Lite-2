"""
DQN Agent for URLLC Dual-Mode Control
Author: Mohammed Hefzi Sobh
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List, Optional
import gymnasium as gym
import sys
sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/src')


class QNetwork(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for URLLC Control
    
    Implements:
    - Double DQN
    - Experience Replay
    - Target Network
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        hidden_dims: List[int] = [64, 64],
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 100,
        device: str = 'auto'
    ):
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Counters
        self.step_count = 0
        self.episode_count = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """Perform one gradient update"""
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']


def train_dqn(
    env: gym.Env,
    agent: DQNAgent,
    n_episodes: int = 500,
    max_steps: int = 10000,
    verbose: bool = True
) -> List[dict]:
    """Train DQN agent"""
    
    history = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        losses = []
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store and update
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        agent.episode_count += 1
        
        # Get metrics
        metrics = env.get_metrics()
        metrics['episode'] = episode
        metrics['episode_reward'] = episode_reward
        metrics['epsilon'] = agent.epsilon
        metrics['avg_loss'] = np.mean(losses) if losses else 0
        history.append(metrics)
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Miss: {metrics['miss_probability']*100:.2f}% | "
                  f"Savings: {metrics['energy_savings']:.1f}% | "
                  f"Eps: {agent.epsilon:.3f}")
    
    return history


def evaluate_dqn(
    env: gym.Env,
    agent: DQNAgent,
    n_episodes: int = 20,
    verbose: bool = True
) -> dict:
    """Evaluate trained DQN agent"""
    
    all_metrics = []
    
    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        
        for step in range(env.config.episode_length):
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            
            if terminated or truncated:
                break
        
        metrics = env.get_metrics()
        all_metrics.append(metrics)
        
        if verbose:
            print(f"Eval {episode+1}/{n_episodes} | "
                  f"Miss: {metrics['miss_probability']*100:.3f}% | "
                  f"Savings: {metrics['energy_savings']:.1f}% | "
                  f"Sw/kTTI: {metrics['switches_per_ktti']:.1f}")
    
    # Aggregate
    result = {
        'miss_probability_mean': np.mean([m['miss_probability'] for m in all_metrics]),
        'miss_probability_std': np.std([m['miss_probability'] for m in all_metrics]),
        'energy_savings_mean': np.mean([m['energy_savings'] for m in all_metrics]),
        'energy_savings_std': np.std([m['energy_savings'] for m in all_metrics]),
        'switches_per_ktti_mean': np.mean([m['switches_per_ktti'] for m in all_metrics]),
        'switches_per_ktti_std': np.std([m['switches_per_ktti'] for m in all_metrics]),
    }
    
    return result


if __name__ == '__main__':
    from environment.urllc_env import URLLCEnv, URRLCConfig
    
    print("="*60)
    print("DQN Agent Training for URLLC Control")
    print("="*60)
    
    # Create environment
    config = URRLCConfig(load=0.85, deadline=0.5, episode_length=10000)
    env = URLLCEnv(config)
    
    # Create agent
    agent = DQNAgent(
        state_dim=5,
        action_dim=2,
        hidden_dims=[64, 64],
        lr=1e-3,
        gamma=0.99,
        epsilon_decay=0.995
    )
    
    print(f"\nDevice: {agent.device}")
    print(f"Training for 200 episodes...")
    
    # Train
    history = train_dqn(env, agent, n_episodes=200, verbose=True)
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluation (20 seeds)")
    print("="*60)
    
    results = evaluate_dqn(env, agent, n_episodes=20)
    
    print(f"\n[DQN Results]")
    print(f"  Miss Prob: {results['miss_probability_mean']*100:.3f}% +/- {results['miss_probability_std']*100:.3f}%")
    print(f"  Savings:   {results['energy_savings_mean']:.1f}% +/- {results['energy_savings_std']:.1f}%")
    print(f"  Sw/kTTI:   {results['switches_per_ktti_mean']:.1f} +/- {results['switches_per_ktti_std']:.1f}")
    
    # Save model
    agent.save('C:/Users/LOQ/Desktop/SENTRY-Lite-2/experiments/results/dqn_model.pt')
    print("\nModel saved!")
