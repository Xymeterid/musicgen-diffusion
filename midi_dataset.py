
import torch
import os
from torch.utils.data import DataLoader, Dataset

from data_conversion import midi_to_event_sequence, events_to_token_sequence


class MIDIDataset(Dataset):
    def __init__(self, data_root, saved_data_path):
        self.sequences = []

        if saved_data_path is not None:
            self.load_dataset(saved_data_path)

        if data_root is not None:
            for filename in os.listdir(data_root):
                filepath = os.path.join(data_root, filename)
                event_sequence = midi_to_event_sequence(filepath)
                token_sequence = events_to_token_sequence(event_sequence)
                self.sequences.append(torch.tensor(token_sequence, dtype=torch.long))

        self.max_length = max(len(seq) for seq in self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq = seq.unsqueeze(-1)
        padded_seq = torch.nn.functional.pad(seq, (0, 0, 0, self.max_length - len(seq)), value=0)
        transposed_seq = padded_seq.transpose(0, 1)
        return transposed_seq

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
