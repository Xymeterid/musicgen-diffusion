import numpy as np


def average_pitch_interval(pitches):
    if len(pitches) < 2:
        return 0
    intervals = [abs(pitches[i] - pitches[i + 1]) for i in range(len(pitches) - 1)]
    return np.mean(intervals)


def mean_pitch_interval(pitches_list):
    pitch_intervals = [average_pitch_interval(pitches) for pitches in pitches_list]
    mean_interval = np.mean(pitch_intervals)
    return mean_interval
