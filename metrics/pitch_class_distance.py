import numpy as np
from scipy.spatial.distance import jensenshannon

def get_note_pitch_class(note_code):
    return note_code % 12


def compute_pitch_class_histogram(pitch_classes, num_classes=12):
    hist, _ = np.histogram(pitch_classes, bins=np.arange(num_classes + 1), density=True)
    return np.array(hist)


def calculate_pitch_class_distance_histogram(pitches_list):
    pitch_classes = [get_note_pitch_class(x) for x in pitches_list]
    return compute_pitch_class_histogram(pitch_classes)


def pitch_class_histogram_distances(generated_histograms, real_histograms):
    distances = [jensenshannon(real_hist, gen_hist) for real_hist, gen_hist in zip(real_histograms, generated_histograms)]
    return np.mean(distances)


def calculate_pitch_class_distances(generated_pitches_lists, real_pitches_lists):
    generated_histograms = [calculate_pitch_class_distance_histogram(generated_pitches_list) for generated_pitches_list
                            in generated_pitches_lists]
    real_histograms = [calculate_pitch_class_distance_histogram(real_pitches_list) for real_pitches_list
                            in real_pitches_lists]

    return pitch_class_histogram_distances(generated_histograms, real_histograms)