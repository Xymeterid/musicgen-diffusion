import random

import numpy as np


def select_random_slice(pitch_list, slice_length):
    if len(pitch_list) < slice_length:
        return pitch_list

    start_index = random.randint(0, len(pitch_list) - slice_length)
    return pitch_list[start_index:start_index + slice_length]

def mean_weighted_pitch_count(pitches_list):
    pitches_list = [select_random_slice(pitch_list, 100) for pitch_list in pitches_list]
    pitch_counts = [len(set(pitches)) / len(pitches) for pitches in pitches_list]
    mean_count = np.mean(pitch_counts)
    return mean_count
