import cv2
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_frequency_analysis(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Module: Normalized 2D FFT Frequency Forensics
    Measures 2D Fast Fourier Transform spectral energy distribution and periodic peak anomalies.
    Outputs frequency_anomaly_score (0.0 to 1.0).
    CRITICAL: This score represents frequency domain deviation, NOT a direct AI probability.
    """
    empty_result = {
        "frequency_anomaly_score": 0.0,
        "low_freq_energy": 0.0,
        "mid_freq_energy": 0.0,
        "high_freq_energy": 0.0,
        "spectral_ratio": 0.0,
        "spectral_entropy": 0.0,
        "radial_slope": 0.0,
        "periodic_peaks_found": False,
        "frequency_score": 0.0,
        "is_spectral_anomaly": False,
    }

    if image_bgr is None or image_bgr.size == 0:
        return empty_result

    try:
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr.copy()

        h, w = gray.shape
        if h > 512 or w > 512:
            gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
            h, w = 512, 512

        # 1. 2D FFT & Shifted Magnitude Spectrum
        fft = np.fft.fft2(gray.astype(float))
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = np.sqrt(cx ** 2 + cy ** 2)

        total_energy = float(np.sum(magnitude ** 2))
        if total_energy == 0:
            return empty_result

        # 2. Frequency Band Energies
        low_mask = r <= (0.15 * max_r)
        mid_mask = (r > 0.15 * max_r) & (r <= 0.50 * max_r)
        high_mask = r > 0.50 * max_r

        low_energy = float(np.sum((magnitude * low_mask) ** 2)) / total_energy
        mid_energy = float(np.sum((magnitude * mid_mask) ** 2)) / total_energy
        high_energy = float(np.sum((magnitude * high_mask) ** 2)) / total_energy

        spectral_ratio = float(high_energy / max(low_energy, 1e-6))

        # 3. Normalized Spectral Entropy
        prob_dist = magnitude / np.sum(magnitude)
        prob_dist = prob_dist[prob_dist > 0]
        spectral_entropy = -float(np.sum(prob_dist * np.log2(prob_dist))) / float(np.log2(magnitude.size))

        # 4. Periodic Grid Peak Analysis (High-frequency spectral spikes typical of generative upsampling)
        magnitude_log = np.log1p(magnitude)
        high_pass_mag = magnitude_log * high_mask
        high_mean = np.mean(high_pass_mag)
        high_std = np.std(high_pass_mag)
        peaks = high_pass_mag > (high_mean + 4.5 * high_std)
        periodic_peaks_found = bool(np.sum(peaks) > 15)

        # 5. Radial Spectrum Falloff Slope
        # Real photographs typically follow ~1/f^alpha (log power vs log freq slope ~ -2.0)
        r_flat = r.ravel()
        mag_flat = magnitude_log.ravel()
        valid_idx = (r_flat > 5) & (r_flat < 0.7 * max_r) & (mag_flat > 0)
        if np.sum(valid_idx) > 100:
            log_r = np.log(r_flat[valid_idx])
            log_m = np.log(mag_flat[valid_idx] + 1e-6)
            slope = float(np.polyfit(log_r, log_m, 1)[0])
        else:
            slope = -2.0

        # Compute Normalized Frequency Anomaly Score (0.0 to 1.0)
        anomaly_score = 0.0

        if high_energy > 0.08:
            anomaly_score += 0.30
        elif high_energy > 0.05:
            anomaly_score += 0.15

        if spectral_ratio > 0.10:
            anomaly_score += 0.30
        elif spectral_ratio > 0.06:
            anomaly_score += 0.15

        if periodic_peaks_found:
            anomaly_score += 0.25

        if abs(slope + 2.0) > 0.8:
            anomaly_score += 0.15

        frequency_anomaly_score = round(min(1.0, anomaly_score), 4)

        return {
            "frequency_anomaly_score": frequency_anomaly_score,
            "low_freq_energy": round(low_energy, 4),
            "mid_freq_energy": round(mid_energy, 4),
            "high_freq_energy": round(high_energy, 4),
            "spectral_ratio": round(spectral_ratio, 4),
            "spectral_entropy": round(spectral_entropy, 4),
            "radial_slope": round(slope, 3),
            "periodic_peaks_found": periodic_peaks_found,
            "frequency_score": frequency_anomaly_score,  # Alias for backward compatibility
            "is_spectral_anomaly": frequency_anomaly_score >= 0.45,
        }
    except Exception as e:
        logger.error(f"Error computing normalized frequency analysis: {e}")
        return empty_result
