"""
URLLC Dual-Mode Control Environment
OpenAI Gym compatible environment for RL training
Author: Mohammed Hefzi Sobh
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class URRLCConfig:
    """Configuration for URLLC environment"""
    # Service capacities
    S_BASE: int = 4          # packets/TTI in BASE mode
    S_BOOST: int = 10        # packets/TTI in BOOST mode
    
    # Timing
    TTI_duration: float = 0.125  # ms
    deadline: float = 0.5        # ms (4 TTIs)
    
    # Traffic (Markov-modulated)
    load: float = 0.85
    burst_factor: float = 3.0
    p_good_to_bad: float = 0.05
    p_bad_to_good: float = 0.3
    
    # Queue
    max_queue: int = 100
    
    # Episode
    episode_length: int = 10000  # TTIs


class MarkovTrafficGenerator:
    """Two-state Markov-modulated traffic"""
    
    def __init__(self, config: URRLCConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.state = 'good'  # 'good' or 'bad'
        
        # Compute arrival rates
        base_rate = config.load * config.S_BASE
        self.good_rate = base_rate * 0.8
        self.bad_rate = base_rate * config.burst_factor
    
    def step(self) -> int:
        """Generate arrivals for one TTI"""
        # State transition
        if self.state == 'good':
            if self.rng.random() < self.config.p_good_to_bad:
                self.state = 'bad'
        else:
            if self.rng.random() < self.config.p_bad_to_good:
                self.state = 'good'
        
        # Generate arrivals
        rate = self.good_rate if self.state == 'good' else self.bad_rate
        return self.rng.poisson(rate)
    
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = 'good'


class URLLCEnv(gym.Env):
    """
    URLLC Dual-Mode Control Environment
    
    Observation Space:
        - queue_length: Current queue size (normalized)
        - avg_delay: Average delay of packets in queue (normalized)
        - current_mode: 0=BASE, 1=BOOST
        - slack: Normalized slack to deadline
        - traffic_state: Estimated traffic intensity
    
    Action Space:
        - 0: Stay in / Switch to BASE
        - 1: Stay in / Switch to BOOST
    
    Reward:
        - Energy reward: +1 for each TTI in BASE
        - Miss penalty: -100 for each missed packet
        - Switch penalty: -1 for each mode switch
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Optional[URRLCConfig] = None, seed: int = 42):
        super().__init__()
        
        self.config = config or URRLCConfig()
        self.seed_value = seed
        
        # Compute deadline in TTIs
        self.deadline_ttis = int(self.config.deadline / self.config.TTI_duration)
        
        # Action space: 0=BASE, 1=BOOST
        self.action_space = spaces.Discrete(2)
        
        # Observation space: [queue, delay, mode, slack, traffic]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize
        self.traffic_gen = MarkovTrafficGenerator(self.config, seed)
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.seed_value = seed
        
        self.traffic_gen.reset(self.seed_value)
        
        # State variables
        self.queue = []  # List of (arrival_time, packet_id)
        self.current_mode = 1  # Start in BOOST for safety
        self.current_step = 0
        self.packet_id = 0
        
        # Metrics
        self.total_arrivals = 0
        self.total_misses = 0
        self.total_served = 0
        self.boost_ttis = 0
        self.switches = 0
        self.last_mode = 1
        
        # EWMA for slack estimation
        self.ewma_slack = 1.0
        self.ewma_alpha = 0.2
        
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        
        # Track mode switch
        if action != self.last_mode:
            self.switches += 1
        self.last_mode = action
        self.current_mode = action
        
        # Track BOOST usage
        if action == 1:
            self.boost_ttis += 1
        
        # Get service capacity
        capacity = self.config.S_BOOST if action == 1 else self.config.S_BASE
        
        # Generate arrivals
        arrivals = self.traffic_gen.step()
        self.total_arrivals += arrivals
        
        # Add packets to queue
        for _ in range(arrivals):
            self.queue.append((self.current_step, self.packet_id))
            self.packet_id += 1
        
        # Check for deadline violations (before serving)
        misses = 0
        new_queue = []
        for arrival_time, pid in self.queue:
            wait_time = self.current_step - arrival_time
            if wait_time > self.deadline_ttis:
                misses += 1
                self.total_misses += 1
            else:
                new_queue.append((arrival_time, pid))
        self.queue = new_queue
        
        # Serve packets (FIFO)
        served = min(capacity, len(self.queue))
        self.queue = self.queue[served:]
        self.total_served += served
        
        # Update EWMA slack
        if len(self.queue) > 0:
            oldest_wait = self.current_step - self.queue[0][0]
            current_slack = max(0, (self.deadline_ttis - oldest_wait) / self.deadline_ttis)
        else:
            current_slack = 1.0
        self.ewma_slack = self.ewma_alpha * current_slack + (1 - self.ewma_alpha) * self.ewma_slack
        
        # Compute reward
        reward = self._compute_reward(action, misses)
        
        # Check termination
        terminated = self.current_step >= self.config.episode_length
        truncated = False
        
        # Info dict
        info = {
            'queue_length': len(self.queue),
            'misses': misses,
            'total_misses': self.total_misses,
            'miss_prob': self.total_misses / max(1, self.total_arrivals),
            'energy_savings': 100 * (1 - self.boost_ttis / self.current_step),
            'switches_per_ktti': 1000 * self.switches / self.current_step,
            'mode': action
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get normalized observation"""
        # Queue length (normalized)
        queue_norm = min(1.0, len(self.queue) / self.config.max_queue)
        
        # Average delay (normalized)
        if len(self.queue) > 0:
            avg_wait = np.mean([self.current_step - t for t, _ in self.queue])
            delay_norm = min(1.0, avg_wait / self.deadline_ttis)
        else:
            delay_norm = 0.0
        
        # Current mode
        mode_norm = float(self.current_mode)
        
        # Slack (EWMA)
        slack_norm = self.ewma_slack
        
        # Traffic intensity estimate
        recent_arrivals = min(len(self.queue), 10)
        traffic_norm = min(1.0, recent_arrivals / self.config.S_BOOST)
        
        return np.array([queue_norm, delay_norm, mode_norm, slack_norm, traffic_norm], dtype=np.float32)
    
    def _compute_reward(self, action: int, misses: int) -> float:
        """Compute reward for this step"""
        # Energy reward: +1 for BASE mode
        energy_reward = 1.0 if action == 0 else 0.0
        
        # Miss penalty: -100 per miss
        miss_penalty = -100.0 * misses
        
        # Switch penalty: -1 per switch
        switch_penalty = -1.0 if (action != self.last_mode and self.current_step > 1) else 0.0
        
        return energy_reward + miss_penalty + switch_penalty
    
    def get_metrics(self) -> Dict[str, float]:
        """Get final performance metrics"""
        return {
            'total_arrivals': self.total_arrivals,
            'total_misses': self.total_misses,
            'total_served': self.total_served,
            'miss_probability': self.total_misses / max(1, self.total_arrivals),
            'energy_savings': 100 * (1 - self.boost_ttis / max(1, self.current_step)),
            'switches_per_ktti': 1000 * self.switches / max(1, self.current_step),
            'boost_fraction': self.boost_ttis / max(1, self.current_step)
        }


# Test
if __name__ == '__main__':
    print("Testing URLLC Environment...")
    
    env = URLLCEnv()
    obs, info = env.reset(seed=42)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial obs: {obs}")
    
    # Run random policy
    total_reward = 0
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    
    metrics = env.get_metrics()
    print(f"\nRandom Policy Results:")
    print(f"  Miss Prob: {metrics['miss_probability']*100:.2f}%")
    print(f"  Energy Savings: {metrics['energy_savings']:.1f}%")
    print(f"  Switches/kTTI: {metrics['switches_per_ktti']:.1f}")
    print(f"  Total Reward: {total_reward:.1f}")
    
    # Run always-BOOST
    obs, _ = env.reset(seed=42)
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(1)  # Always BOOST
        if terminated:
            break
    
    metrics = env.get_metrics()
    print(f"\nAlways-BOOST Results:")
    print(f"  Miss Prob: {metrics['miss_probability']*100:.2f}%")
    print(f"  Energy Savings: {metrics['energy_savings']:.1f}%")
    
    print("\n✓ Environment test passed!")
