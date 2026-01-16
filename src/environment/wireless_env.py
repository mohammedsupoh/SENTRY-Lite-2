"""
W1 SOLUTION: Enhanced Wireless-Grounded Environment
Adds realistic wireless channel effects to URLLC simulation

Key additions:
1. Rayleigh/Rician fading channel
2. BLER-based packet errors
3. CQI-dependent capacity
4. HARQ retransmissions
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum

class ChannelModel(Enum):
    AWGN = "awgn"
    RAYLEIGH = "rayleigh"
    RICIAN = "rician"
    EPA = "epa"  # Extended Pedestrian A
    EVA = "eva"  # Extended Vehicular A

@dataclass
class WirelessConfig:
    """3GPP-aligned wireless configuration"""
    # Channel
    channel_model: ChannelModel = ChannelModel.RAYLEIGH
    snr_db: float = 10.0  # Average SNR
    rician_k: float = 3.0  # Rician K-factor (for LOS)
    doppler_hz: float = 10.0  # Doppler spread (mobility)
    
    # PHY/MAC
    tti_ms: float = 0.125  # Mini-slot for URLLC (125 μs)
    prb_count: int = 10  # PRBs allocated to UE
    mcs_table: str = "qam64"  # MCS table
    harq_max_retx: int = 1  # Max HARQ retransmissions (URLLC: 0-1)
    
    # BLER targets
    target_bler: float = 1e-5  # URLLC BLER target
    
    # Capacity (packets per TTI)
    capacity_base: int = 4
    capacity_boost: int = 10


class WirelessChannel:
    """Realistic wireless channel model"""
    
    def __init__(self, config: WirelessConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Channel state
        self.current_snr = config.snr_db
        self.current_cqi = self._snr_to_cqi(config.snr_db)
        
        # Fading state (for correlated fading)
        self._fading_state = 1.0
        
    def _snr_to_cqi(self, snr_db: float) -> int:
        """Map SNR to CQI (simplified 3GPP mapping)"""
        # Approximate CQI-SNR mapping from 3GPP TS 38.214
        cqi_thresholds = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 
                          8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7]
        for cqi, thresh in enumerate(cqi_thresholds, 1):
            if snr_db < thresh:
                return max(1, cqi - 1)
        return 15
    
    def _cqi_to_bler(self, cqi: int, mode: int) -> float:
        """Estimate BLER based on CQI and transmission mode"""
        # Simplified BLER model
        # Higher mode (BOOST) uses more aggressive MCS → higher BLER at same CQI
        base_bler = {
            1: 0.1, 2: 0.08, 3: 0.05, 4: 0.03, 5: 0.02,
            6: 0.01, 7: 0.005, 8: 0.002, 9: 0.001, 10: 0.0005,
            11: 0.0002, 12: 0.0001, 13: 0.00005, 14: 0.00002, 15: 0.00001
        }
        bler = base_bler.get(cqi, 0.1)
        
        # BOOST mode: slightly higher BLER due to aggressive scheduling
        if mode == 1:  # BOOST
            bler *= 1.5
        
        return min(bler, 0.3)
    
    def update_channel(self) -> Tuple[float, int]:
        """Update channel state (call each TTI)"""
        config = self.config
        
        if config.channel_model == ChannelModel.AWGN:
            # No fading
            self.current_snr = config.snr_db
            
        elif config.channel_model == ChannelModel.RAYLEIGH:
            # Rayleigh fading (NLOS)
            # Jake's model for temporal correlation
            fd_normalized = config.doppler_hz * (config.tti_ms / 1000)
            correlation = np.exp(-2 * np.pi * fd_normalized)
            
            # Generate correlated fading
            innovation = self.rng.standard_normal()
            self._fading_state = (correlation * self._fading_state + 
                                  np.sqrt(1 - correlation**2) * innovation)
            
            # Convert to power (exponential distribution for Rayleigh)
            fading_power = np.abs(self._fading_state)**2
            self.current_snr = config.snr_db + 10 * np.log10(fading_power + 1e-10)
            
        elif config.channel_model == ChannelModel.RICIAN:
            # Rician fading (LOS component)
            k = config.rician_k
            fd_normalized = config.doppler_hz * (config.tti_ms / 1000)
            correlation = np.exp(-2 * np.pi * fd_normalized)
            
            innovation = self.rng.standard_normal()
            self._fading_state = (correlation * self._fading_state + 
                                  np.sqrt(1 - correlation**2) * innovation)
            
            # Rician: LOS + scattered
            los_component = np.sqrt(k / (k + 1))
            scatter_component = np.sqrt(1 / (k + 1)) * self._fading_state
            fading_power = np.abs(los_component + scatter_component)**2
            self.current_snr = config.snr_db + 10 * np.log10(fading_power + 1e-10)
        
        self.current_cqi = self._snr_to_cqi(self.current_snr)
        return self.current_snr, self.current_cqi
    
    def get_effective_capacity(self, mode: int) -> int:
        """Get capacity based on current CQI and mode"""
        base_cap = self.config.capacity_base if mode == 0 else self.config.capacity_boost
        
        # CQI-based capacity scaling (simplified)
        cqi_factor = self.current_cqi / 10.0  # Normalize to ~1.0 at CQI=10
        effective_cap = int(base_cap * cqi_factor)
        
        return max(1, effective_cap)
    
    def transmit_packet(self, mode: int) -> bool:
        """Simulate packet transmission with BLER"""
        bler = self._cqi_to_bler(self.current_cqi, mode)
        
        # First transmission
        if self.rng.random() > bler:
            return True  # Success
        
        # HARQ retransmission (if allowed)
        if self.config.harq_max_retx > 0:
            # Retransmission has better BLER (combining gain)
            retx_bler = bler * 0.1  # ~10x improvement
            if self.rng.random() > retx_bler:
                return True
        
        return False  # Packet lost


class WirelessURLLCEnv:
    """
    URLLC Environment with Wireless Channel Effects
    
    This addresses TWC reviewer concern W1:
    - Realistic channel model (Rayleigh/Rician fading)
    - CQI-dependent capacity
    - BLER-based packet errors
    - HARQ retransmissions
    """
    
    def __init__(self, config: WirelessConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.channel = WirelessChannel(config, seed)
        
        # Queue state
        self.queue = []
        self.current_mode = 1  # Start in BOOST
        self.time = 0
        
        # Metrics
        self.total_arrivals = 0
        self.total_departures = 0
        self.deadline_misses = 0
        self.packet_errors = 0  # Due to BLER
        self.mode_switches = 0
        self.base_ttis = 0
        
        # Deadline (in TTIs)
        self.deadline_ttis = int(0.5 / config.tti_ms)  # 0.5ms deadline
        
        # EWMA slack
        self.ewma_slack = 0.5
        self.ewma_alpha = 0.1
        
    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.channel = WirelessChannel(self.config, seed)
        
        self.queue = []
        self.current_mode = 1
        self.time = 0
        self.total_arrivals = 0
        self.total_departures = 0
        self.deadline_misses = 0
        self.packet_errors = 0
        self.mode_switches = 0
        self.base_ttis = 0
        self.ewma_slack = 0.5
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """State: [queue_norm, slack, mode, cqi_norm, snr_norm]"""
        q_norm = min(len(self.queue) / 30.0, 1.0)
        cqi_norm = self.channel.current_cqi / 15.0
        snr_norm = (self.channel.current_snr + 10) / 40.0  # Normalize [-10, 30] to [0, 1]
        
        return np.array([
            q_norm,
            self.ewma_slack,
            float(self.current_mode),
            cqi_norm,
            snr_norm
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one TTI"""
        
        # 1. Update channel
        snr, cqi = self.channel.update_channel()
        
        # 2. Mode switch
        if action != self.current_mode:
            self.mode_switches += 1
            self.current_mode = action
        
        if self.current_mode == 0:
            self.base_ttis += 1
        
        # 3. Arrivals (Poisson)
        load = 0.85
        mean_arrivals = load * self.config.capacity_base
        arrivals = self.rng.poisson(mean_arrivals)
        
        for _ in range(arrivals):
            self.queue.append(self.deadline_ttis)
            self.total_arrivals += 1
        
        # 4. Service with BLER
        capacity = self.channel.get_effective_capacity(self.current_mode)
        served = 0
        
        for _ in range(min(capacity, len(self.queue))):
            if self.channel.transmit_packet(self.current_mode):
                self.queue.pop(0)
                self.total_departures += 1
                served += 1
            else:
                self.packet_errors += 1
        
        # 5. Age packets and check deadlines
        new_queue = []
        for remaining in self.queue:
            if remaining <= 1:
                self.deadline_misses += 1
            else:
                new_queue.append(remaining - 1)
        self.queue = new_queue
        
        # 6. Update slack
        if len(self.queue) > 0:
            min_remaining = min(self.queue)
            slack = min_remaining / self.deadline_ttis
        else:
            slack = 1.0
        self.ewma_slack = self.ewma_alpha * slack + (1 - self.ewma_alpha) * self.ewma_slack
        
        self.time += 1
        
        # Info
        info = {
            'snr': snr,
            'cqi': cqi,
            'capacity': capacity,
            'served': served,
            'queue_len': len(self.queue),
            'packet_errors': self.packet_errors
        }
        
        return self._get_observation(), 0.0, False, info
    
    def get_metrics(self) -> dict:
        """Get performance metrics"""
        miss_prob = self.deadline_misses / max(1, self.total_arrivals)
        error_rate = self.packet_errors / max(1, self.total_arrivals)
        energy_savings = self.base_ttis / max(1, self.time) * 100
        switches_per_ktti = self.mode_switches / max(1, self.time) * 1000
        
        return {
            'miss_probability': miss_prob,
            'packet_error_rate': error_rate,
            'energy_savings': energy_savings,
            'switches_per_ktti': switches_per_ktti,
            'total_arrivals': self.total_arrivals,
            'deadline_misses': self.deadline_misses,
            'packet_errors': self.packet_errors
        }


#=============================================================================
# TEST
#=============================================================================

if __name__ == '__main__':
    print("="*70)
    print("WIRELESS-GROUNDED URLLC ENVIRONMENT TEST")
    print("="*70)
    
    # Test different channel models
    for channel_model in [ChannelModel.AWGN, ChannelModel.RAYLEIGH, ChannelModel.RICIAN]:
        print(f"\n[Channel: {channel_model.value}]")
        print("-"*50)
        
        config = WirelessConfig(
            channel_model=channel_model,
            snr_db=15.0,
            doppler_hz=10.0
        )
        
        results = {'miss': [], 'error': [], 'savings': [], 'switches': []}
        
        for seed in range(5):
            env = WirelessURLLCEnv(config, seed=seed)
            env.reset(seed=seed)
            
            # Run SENTRY-Lite-2 policy
            counter = 0
            for t in range(10000):
                Q = len(env.queue)
                slack = env.ewma_slack
                cqi = env.channel.current_cqi
                
                # SENTRY-Lite-2 with CQI awareness
                if Q >= 15 or slack < 0.25 or cqi < 8:
                    action = 1  # BOOST
                    counter = 0
                elif Q < 6 and slack > 0.50 and cqi >= 10:
                    counter += 1
                    if counter >= 2:
                        action = 0  # BASE
                    else:
                        action = env.current_mode
                else:
                    action = env.current_mode
                    counter = 0
                
                env.step(action)
            
            metrics = env.get_metrics()
            results['miss'].append(metrics['miss_probability'] * 100)
            results['error'].append(metrics['packet_error_rate'] * 100)
            results['savings'].append(metrics['energy_savings'])
            results['switches'].append(metrics['switches_per_ktti'])
        
        print(f"  Miss Prob:    {np.mean(results['miss']):.3f}% ± {np.std(results['miss']):.3f}%")
        print(f"  Packet Error: {np.mean(results['error']):.3f}% ± {np.std(results['error']):.3f}%")
        print(f"  Savings:      {np.mean(results['savings']):.1f}% ± {np.std(results['savings']):.1f}%")
        print(f"  Sw/kTTI:      {np.mean(results['switches']):.1f} ± {np.std(results['switches']):.1f}")
    
    print("\n" + "="*70)
    print("W1 ADDRESSED: Environment now includes:")
    print("  ✓ Rayleigh/Rician fading channel")
    print("  ✓ CQI-dependent capacity")
    print("  ✓ BLER-based packet errors")
    print("  ✓ HARQ retransmissions")
    print("="*70)
