"""
Diagnostic: Why is URLLC failing even with Always-BOOST under fading?
"""

import numpy as np
import sys
sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/src')
from environment.wireless_env import WirelessURLLCEnv, WirelessConfig, ChannelModel


def diagnose_channel(channel_model: ChannelModel, snr_db: float = 15.0):
    """Diagnose what's happening with the channel"""
    
    config = WirelessConfig(
        channel_model=channel_model,
        snr_db=snr_db,
        doppler_hz=10.0
    )
    
    env = WirelessURLLCEnv(config, seed=42)
    env.reset(seed=42)
    
    # Collect statistics
    cqi_history = []
    snr_history = []
    capacity_history = []
    served_history = []
    queue_history = []
    
    # Always BOOST
    for t in range(10000):
        cqi_history.append(env.channel.current_cqi)
        snr_history.append(env.channel.current_snr)
        
        Q = len(env.queue)
        queue_history.append(Q)
        
        # Get capacity before step
        cap = env.channel.get_effective_capacity(1)  # BOOST mode
        capacity_history.append(cap)
        
        obs, _, _, info = env.step(1)  # Always BOOST
        served_history.append(info.get('served', 0))
    
    metrics = env.get_metrics()
    
    print(f"\n[Channel: {channel_model.value}, SNR={snr_db} dB]")
    print("-"*50)
    print(f"  CQI:      min={min(cqi_history):2d}, max={max(cqi_history):2d}, mean={np.mean(cqi_history):.1f}")
    print(f"  SNR:      min={min(snr_history):.1f}, max={max(snr_history):.1f}, mean={np.mean(snr_history):.1f} dB")
    print(f"  Capacity: min={min(capacity_history):2d}, max={max(capacity_history):2d}, mean={np.mean(capacity_history):.1f}")
    print(f"  Served:   min={min(served_history):2d}, max={max(served_history):2d}, mean={np.mean(served_history):.1f}")
    print(f"  Queue:    min={min(queue_history):2d}, max={max(queue_history):2d}, mean={np.mean(queue_history):.1f}")
    print(f"  Miss Prob (Always BOOST): {metrics['miss_probability']*100:.3f}%")
    
    # Check: what fraction of time is capacity < arrivals?
    load = 0.85
    arrival_rate = load * 4  # 3.4 packets/TTI on average
    undercapacity_frac = sum(1 for c in capacity_history if c < arrival_rate) / len(capacity_history)
    print(f"  Time with capacity < arrival_rate: {undercapacity_frac*100:.1f}%")
    
    return metrics['miss_probability']


def find_required_snr(channel_model: ChannelModel, target_miss: float = 0.01):
    """Find SNR required for target miss probability"""
    
    print(f"\n{'='*50}")
    print(f"Finding required SNR for {target_miss*100:.1f}% miss under {channel_model.value}")
    print(f"{'='*50}")
    
    for snr in [15, 18, 20, 22, 25, 28, 30]:
        miss = diagnose_channel(channel_model, snr)
        if miss < target_miss:
            print(f"\n  ✓ SNR={snr} dB achieves target!")
            return snr
    
    print(f"\n  ✗ Could not achieve target even at SNR=30 dB")
    return None


if __name__ == '__main__':
    print("="*60)
    print("DIAGNOSTIC: CHANNEL IMPACT ON URLLC PERFORMANCE")
    print("="*60)
    
    # 1. Diagnose current settings
    print("\n[1. Current Settings (SNR=15 dB)]")
    for ch in [ChannelModel.AWGN, ChannelModel.RAYLEIGH, ChannelModel.RICIAN]:
        diagnose_channel(ch, 15.0)
    
    # 2. Find required SNR
    print("\n[2. Finding Required SNR for URLLC (<1% miss)]")
    for ch in [ChannelModel.RAYLEIGH, ChannelModel.RICIAN]:
        find_required_snr(ch, 0.01)
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("  If required SNR >> 15 dB, the wireless model may be too harsh")
    print("  or we need to revisit the capacity/BLER mapping")
    print("="*60)
