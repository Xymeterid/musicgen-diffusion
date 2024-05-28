import numpy as np
from scipy.special import erf


def get_pitches(notes):
    notes.sort(key=lambda x: x[0])
    pitches = [note[2] for note in notes]
    return pitches


def sliding_window_statistics(pitches, window_size, hop_size):
    num_windows = (len(pitches) - window_size) // hop_size + 1
    pitch_stats = []
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        window_pitches = pitches[start:end]

        pitch_mean = np.mean(window_pitches)
        pitch_variance = np.var(window_pitches)

        pitch_stats.append((pitch_mean, pitch_variance))
    return pitch_stats


def overlapping_area(mu1, sigma1, mu2, sigma2):
    c = (mu1 + mu2) / 2
    oa = 1 - erf((c - mu1) / np.sqrt(2 * sigma1)) + erf((c - mu2) / np.sqrt(2 * sigma2))
    return oa


def compute_oa_stats(original_stats, generated_stats):
    oas = []
    for (mu1, sigma1), (mu2, sigma2) in zip(original_stats, generated_stats):
        oa = overlapping_area(mu1, sigma1, mu2, sigma2)
        oas.append(oa)
    return np.mean(oas), np.var(oas)


def compute_consistency_variance(mu_oa, sigma_oa, mu_gt, sigma_gt):
    consistency = max(0, 1 - abs(mu_oa - mu_gt) / mu_gt)
    variance = max(0, 1 - abs(sigma_oa - sigma_gt) / sigma_gt)
    return consistency, variance


WINDOW_SIZE = 16  # Adjusted window size (16 notes to represent measures)
HOP_SIZE = 8  # Adjusted hop size (8 notes to represent measures)


def calculate_framewise_similarity(original_notes, generated_notes):
    original_pitches = get_pitches(original_notes)
    generated_pitches = get_pitches(generated_notes)

    original_pitch_stats = sliding_window_statistics(original_pitches, WINDOW_SIZE, HOP_SIZE)
    generated_pitch_stats = sliding_window_statistics(generated_pitches, WINDOW_SIZE, HOP_SIZE)

    original_pitch_oa_mean, original_pitch_oa_var = compute_oa_stats(original_pitch_stats, generated_pitch_stats)

    gt_pitch_oa_mean = np.mean([stat[0] for stat in original_pitch_stats])
    gt_pitch_oa_var = np.var([stat[1] for stat in original_pitch_stats])

    pitch_consistency, pitch_variance = compute_consistency_variance(
        original_pitch_oa_mean, original_pitch_oa_var, gt_pitch_oa_mean, gt_pitch_oa_var)

    return pitch_consistency, pitch_variance
