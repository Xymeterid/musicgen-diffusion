import numpy as np


def pitch_range(pitches):
    if len(pitches) == 0:
        return 0
    return max(pitches) - min(pitches)


def mean_pitch_range(pitches_list):
    pitch_ranges = [pitch_range(pitches) for pitches in pitches_list]
    mean_range = np.mean(pitch_ranges)
    return mean_range
