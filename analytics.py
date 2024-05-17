import numpy as np
from scipy.spatial.distance import jensenshannon

from data_conversion import token_sequence_to_events


def extract_note_list_from_sample(sample):
    sample_events = token_sequence_to_events(sample.toList())
    return [event[1] for event in sample_events if event[0] == 'note_on']


def get_note_pitch_class(note_code):
    return note_code % 12


def compute_pitch_class_histogram(pitch_classes, num_classes=12):
    hist, _ = np.histogram(pitch_classes, bins=np.arange(num_classes + 1), density=True)
    return np.array(hist)


def pitch_class_histogram_distances(generated_histograms, real_histograms):
    distances = [jensenshannon(real_hist, gen_hist) for real_hist, gen_hist in zip(real_histograms, generated_histograms)]
    return np.mean(distances)


def calculate_pitch_class_distance_histogram(sample):
    sample_note_list = extract_note_list_from_sample(sample)
    pitch_classes = [get_note_pitch_class(x) for x in sample_note_list]
    return compute_pitch_class_histogram(pitch_classes)
