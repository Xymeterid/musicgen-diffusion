
import torch
import os
from torch.utils.data import DataLoader, Dataset

from data_conversion import midi_to_event_sequences, events_to_token_sequence, SEQUENCE_DURATION


class MIDIDataset(Dataset):
    def __init__(self, data_root, saved_data_path):
        self.sequences = []

        if saved_data_path is not None:
            self.load_dataset(saved_data_path)

        if data_root is not None:
            for filename in os.listdir(data_root):
                filepath = os.path.join(data_root, filename)
                event_sequences = midi_to_event_sequences(filepath)
                for event_sequence in event_sequences:
                    token_sequence = events_to_token_sequence(event_sequence)
                    tensor = torch.tensor(token_sequence, dtype=torch.float16)
                    tensor = tensor.unsqueeze(-1)
                    tensor = tensor.transpose(0, 1)
                    self.sequences.append(tensor)

        self.max_length = SEQUENCE_DURATION

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def save_dataset(self, file_path):
        torch.save({
            'sequences': self.sequences,
            'max_length': self.max_length
        }, file_path)

    def load_dataset(self, file_path):
        data = torch.load(file_path)
        self.sequences = data['sequences']
        self.max_length = data['max_length']

    def get_data_loader(self, batch_size=32):
        return DataLoader(self, batch_size=batch_size, shuffle=True)
