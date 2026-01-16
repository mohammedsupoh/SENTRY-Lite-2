"""
P1 Extended: Find Multi-UE URLLC boundary at higher SNR
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'C:/Users/LOQ/Desktop/SENTRY-Lite-2/experiments')
from multi_ue_fixed import run_multi_ue_experiment, ChannelModel

if __name__ == '__main__':
    print("="*70)
    print("P1 EXTENDED: FINDING MULTI-UE URLLC BOUNDARY")
    print("="*70)
    
    # Test higher SNR for multi-UE
    ue_counts = [2, 5]
    snrs = [25, 26, 27, 28, 30]
    
    results = []
    
    for n_ues in ue_counts:
        print(f"\n[Testing {n_ues} UEs]")
        for snr in snrs:
            print(f"  SNR={snr} dB...", end=" ")
            
            res = run_multi_ue_experiment(
                n_ues=n_ues,
                channel=ChannelModel.RAYLEIGH,
                snr=snr,
                scheduler='EDF',
                n_seeds=5
            )
            
            urllc = "✓" if res['miss_mean'] < 1.0 else "✗"
            print(f"Miss={res['miss_mean']:.3f}% {urllc}")
            
            results.append({
                'n_ues': n_ues,
                'snr': snr,
                **res
            })
            
            # Stop if URLLC achieved
            if res['miss_mean'] < 1.0:
                print(f"  → URLLC boundary found at SNR={snr} dB for {n_ues} UEs")
                break
    
    print("\n" + "="*70)
    print("[MULTI-UE OPERATIONAL ENVELOPE]")
    print("-"*70)
    
    for n_ues in ue_counts:
        ue_results = [r for r in results if r['n_ues'] == n_ues]
        compliant = [r for r in ue_results if r['miss_mean'] < 1.0]
        if compliant:
            min_snr = min(r['snr'] for r in compliant)
            best = [r for r in compliant if r['snr'] == min_snr][0]
            print(f"  {n_ues} UEs: URLLC @ SNR ≥ {min_snr} dB "
                  f"(Miss={best['miss_mean']:.2f}%, Save={best['savings_mean']:.1f}%)")
        else:
            closest = min(ue_results, key=lambda x: x['miss_mean'])
            print(f"  {n_ues} UEs: Best @ SNR={closest['snr']} dB "
                  f"(Miss={closest['miss_mean']:.2f}%, needs higher SNR)")
    
    print("\n" + "="*70)
