from midi_dataset import MIDIDataset

data_root = 'data/maestro/maestro-v1.0.0/small'
dataset = MIDIDataset(data_root=data_root, saved_data_path=None)
dataset.save_dataset('midi_dataset.pth')
