"""
P3: SNR ↔ CQI ↔ BLER Mapping Table
Based on 3GPP TS 38.214 with simplified BLER model
"""

import numpy as np
import pandas as pd

# CQI to SNR mapping (3GPP TS 38.214 Table 5.2.2.1-3)
# These are minimum SNR thresholds for each CQI
CQI_SNR_THRESHOLDS = {
    1: -6.7,   # QPSK, code rate 78/1024
    2: -4.7,   # QPSK, code rate 120/1024
    3: -2.3,   # QPSK, code rate 193/1024
    4:  0.2,   # QPSK, code rate 308/1024
    5:  2.4,   # QPSK, code rate 449/1024
    6:  4.3,   # QPSK, code rate 602/1024
    7:  5.9,   # 16QAM, code rate 378/1024
    8:  8.1,   # 16QAM, code rate 490/1024
    9: 10.3,   # 16QAM, code rate 616/1024
    10: 11.7,  # 64QAM, code rate 466/1024
    11: 14.1,  # 64QAM, code rate 567/1024
    12: 16.3,  # 64QAM, code rate 666/1024
    13: 18.7,  # 64QAM, code rate 772/1024
    14: 21.0,  # 64QAM, code rate 873/1024
    15: 22.7,  # 64QAM, code rate 948/1024
}

# CQI to MCS mapping (simplified)
CQI_MCS_MAPPING = {
    1: {'modulation': 'QPSK', 'code_rate': 0.076, 'spectral_eff': 0.15},
    2: {'modulation': 'QPSK', 'code_rate': 0.117, 'spectral_eff': 0.23},
    3: {'modulation': 'QPSK', 'code_rate': 0.188, 'spectral_eff': 0.38},
    4: {'modulation': 'QPSK', 'code_rate': 0.301, 'spectral_eff': 0.60},
    5: {'modulation': 'QPSK', 'code_rate': 0.438, 'spectral_eff': 0.88},
    6: {'modulation': 'QPSK', 'code_rate': 0.588, 'spectral_eff': 1.18},
    7: {'modulation': '16QAM', 'code_rate': 0.369, 'spectral_eff': 1.48},
    8: {'modulation': '16QAM', 'code_rate': 0.479, 'spectral_eff': 1.91},
    9: {'modulation': '16QAM', 'code_rate': 0.602, 'spectral_eff': 2.41},
    10: {'modulation': '64QAM', 'code_rate': 0.455, 'spectral_eff': 2.73},
    11: {'modulation': '64QAM', 'code_rate': 0.554, 'spectral_eff': 3.32},
    12: {'modulation': '64QAM', 'code_rate': 0.650, 'spectral_eff': 3.90},
    13: {'modulation': '64QAM', 'code_rate': 0.754, 'spectral_eff': 4.52},
    14: {'modulation': '64QAM', 'code_rate': 0.853, 'spectral_eff': 5.12},
    15: {'modulation': '64QAM', 'code_rate': 0.926, 'spectral_eff': 5.55},
}

# BLER model (simplified - at 10% BLER operating point)
CQI_BLER_MODEL = {
    1: 0.10,    # High BLER at low CQI
    2: 0.08,
    3: 0.05,
    4: 0.03,
    5: 0.02,
    6: 0.01,
    7: 0.005,
    8: 0.002,
    9: 0.001,
    10: 0.0005,
    11: 0.0002,
    12: 0.0001,
    13: 0.00005,
    14: 0.00002,
    15: 0.00001,  # Very low BLER at high CQI
}


def generate_mapping_table():
    """Generate comprehensive SNR↔CQI↔MCS↔BLER table"""
    
    rows = []
    for cqi in range(1, 16):
        mcs = CQI_MCS_MAPPING[cqi]
        rows.append({
            'CQI': cqi,
            'Min_SNR_dB': CQI_SNR_THRESHOLDS[cqi],
            'Modulation': mcs['modulation'],
            'Code_Rate': f"{mcs['code_rate']:.3f}",
            'Spectral_Eff': f"{mcs['spectral_eff']:.2f}",
            'BLER_target': f"{CQI_BLER_MODEL[cqi]:.0e}",
        })
    
    return pd.DataFrame(rows)


def snr_to_expected_cqi(snr_db: float, channel: str = 'awgn') -> dict:
    """Map SNR to expected CQI under different channels"""
    
    # AWGN: deterministic mapping
    if channel == 'awgn':
        for cqi in range(15, 0, -1):
            if snr_db >= CQI_SNR_THRESHOLDS[cqi]:
                return {'cqi': cqi, 'type': 'deterministic'}
        return {'cqi': 1, 'type': 'deterministic'}
    
    # Rayleigh: SNR varies
    elif channel == 'rayleigh':
        median_snr = snr_db - 1.6  # Approximate median shift
        for cqi in range(15, 0, -1):
            if median_snr >= CQI_SNR_THRESHOLDS[cqi]:
                return {'cqi_median': cqi, 'cqi_range': (max(1, cqi-3), min(15, cqi+2)), 'type': 'fading'}
        return {'cqi_median': 1, 'cqi_range': (1, 5), 'type': 'fading'}
    
    # Rician (K=3): less severe fading
    elif channel == 'rician':
        median_snr = snr_db - 0.5
        for cqi in range(15, 0, -1):
            if median_snr >= CQI_SNR_THRESHOLDS[cqi]:
                return {'cqi_median': cqi, 'cqi_range': (max(1, cqi-2), min(15, cqi+1)), 'type': 'fading'}
        return {'cqi_median': 1, 'cqi_range': (1, 4), 'type': 'fading'}


def generate_envelope_context_table():
    """Generate table showing SNR thresholds in context"""
    
    rows = []
    for snr in [15, 18, 20, 22, 25]:
        awgn = snr_to_expected_cqi(snr, 'awgn')
        rayleigh = snr_to_expected_cqi(snr, 'rayleigh')
        rician = snr_to_expected_cqi(snr, 'rician')
        
        rows.append({
            'SNR_avg_dB': snr,
            'AWGN_CQI': awgn['cqi'],
            'Rayleigh_CQI_median': rayleigh['cqi_median'],
            'Rayleigh_CQI_range': f"{rayleigh['cqi_range'][0]}-{rayleigh['cqi_range'][1]}",
            'Rician_CQI_median': rician['cqi_median'],
            'Rician_CQI_range': f"{rician['cqi_range'][0]}-{rician['cqi_range'][1]}",
        })
    
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("="*70)
    print("P3: SNR ↔ CQI ↔ MCS ↔ BLER MAPPING TABLE")
    print("="*70)
    
    # Table 1: Full mapping
    print("\n[Table 1: CQI → SNR → MCS → BLER Mapping (3GPP TS 38.214)]")
    print("-"*70)
    df1 = generate_mapping_table()
    print(df1.to_string(index=False))
    
    # Table 2: Envelope context
    print("\n[Table 2: Average SNR → Expected CQI per Channel]")
    print("-"*70)
    df2 = generate_envelope_context_table()
    print(df2.to_string(index=False))
    
    # Save
    df1.to_csv('experiments/results/cqi_snr_mapping.csv', index=False)
    df2.to_csv('experiments/results/envelope_snr_context.csv', index=False)
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("  At SNR=20 dB (Rayleigh URLLC threshold):")
    print("    - Median CQI = 12 (64QAM, code rate 0.65)")
    print("    - CQI can drop to 9 during deep fades")
    print("    - This explains why P_miss > 0 even at high avg SNR")
    print("="*70)
