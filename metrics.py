import os
import random

import pretty_midi

from metrics.average_pitch_interval import mean_pitch_interval
from metrics.note_count import mean_note_count
from metrics.pitch_class_distance import calculate_pitch_class_distances
from metrics.pitch_range import mean_pitch_range
from metrics.weighted_pitch_count import mean_weighted_pitch_count


def load_midi(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch))
    return notes


def get_pitches(notes):
    notes.sort(key=lambda x: x[0])
    # notes = notes[:100]
    pitches = [note[2] for note in notes]
    return pitches


def load_midi_files(file_paths):
    all_pitches = []
    for file_path in file_paths:
        notes = load_midi(file_path)
        pitches = get_pitches(notes)
        all_pitches.append(pitches)
    return all_pitches


def get_random_dataset_data(dataset_filepath, sample_count=10):
    all_files = [f for f in os.listdir(dataset_filepath) if os.path.isfile(os.path.join(dataset_filepath, f))]
    if len(all_files) < sample_count:
        raise ValueError("Not enough files in the folder to select the desired number of random files.")

    selected_files = random.sample(all_files, sample_count)
    selected_files = [dataset_filepath + file for file in selected_files]

    return selected_files


original_midi_paths = get_random_dataset_data('data/maestro/full/', 20)
generated_midi_paths = get_random_dataset_data('results/samples/', 20)
generated_midi_paths_40 = get_random_dataset_data('results/samples_40/', 20)

original_pitches = load_midi_files(original_midi_paths)
generated_pitches = load_midi_files(generated_midi_paths)
generated_pitches_40 = load_midi_files(generated_midi_paths_40)

original_mean_pitch_count = mean_weighted_pitch_count(original_pitches)
generated_mean_pitch_count = mean_weighted_pitch_count(generated_pitches)
generated_mean_pitch_count_40 = mean_weighted_pitch_count(generated_pitches_40)

print(f"Mean Original Pitch Count: {original_mean_pitch_count}")
print(f"Mean Generated Pitch Count: {generated_mean_pitch_count}")
print(f"Mean Generated Pitch Count 40%: {generated_mean_pitch_count_40}\n")

original_mean_note_count = mean_note_count(original_pitches)
generated_mean_note_count = mean_note_count(generated_pitches)
generated_mean_note_count_40 = mean_note_count(generated_pitches_40)

print(f"Mean Note Count: {original_mean_note_count}")
print(f"Mean Note Count: {generated_mean_note_count}")
print(f"Mean Note Count 40%: {generated_mean_note_count_40}\n")

original_mean_pitch_range = mean_pitch_range(original_pitches)
generated_mean_pitch_range = mean_pitch_range(generated_pitches)
generated_mean_pitch_range_40 = mean_pitch_range(generated_pitches_40)

print(f"Mean Pitch Range: {original_mean_pitch_range}")
print(f"Mean Pitch Range: {generated_mean_pitch_range}")
print(f"Mean Pitch Range 40%: {generated_mean_pitch_range_40}\n")

original_mean_pitch_interval = mean_pitch_interval(original_pitches)
generated_mean_pitch_interval = mean_pitch_interval(generated_pitches)
generated_mean_pitch_interval_40 = mean_pitch_interval(generated_pitches_40)

print(f"Mean Pitch Interval: {original_mean_pitch_interval}")
print(f"Mean Pitch Interval: {generated_mean_pitch_interval}")
print(f"Mean Pitch Interval 40%: {generated_mean_pitch_interval_40}\n")

pitch_class_distances = calculate_pitch_class_distances(original_pitches, generated_pitches)
pitch_class_distances_40 = calculate_pitch_class_distances(original_pitches, generated_pitches_40)

print(f"Pitch Class Distances (less is better): {pitch_class_distances}")
print(f"Pitch Class Distances 40 (less is better): {pitch_class_distances_40}")

# original_notes = load_midi(original_midi_path)
# generated_notes = load_midi(generated_midi_path)

# pitch_consistency, pitch_variance = calculate_framewise_similarity(original_notes, generated_notes)
# print(f"Pitch Consistency: {pitch_consistency}, Pitch Variance: {pitch_variance}")
