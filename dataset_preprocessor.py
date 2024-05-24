from data_conversion import token_sequence_to_events, event_sequence_to_midi
from midi_dataset import MIDIDataset

data_root = 'data/maestro/maestro-v1.0.0/2004'
dataset = MIDIDataset(data_root=data_root, saved_data_path=None)
dataset.save_dataset('midi_dataset_float.pth')

# for index, item in enumerate(dataset):
#     generated_data = item.tolist()[0]
#     events = token_sequence_to_events(generated_data)
#     event_sequence_to_midi(events, f'sample_{index}.midi')