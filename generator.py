import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D

from data_conversion import token_sequence_to_events, event_sequence_to_midi
from midi_dataset import MIDIDataset

dataset = MIDIDataset(data_root=None, saved_data_path='midi_dataset.pth')
model = Unet1D(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length=dataset.max_length,
    timesteps=1000,
    beta_schedule='cosine',
)

diffusion.load_state_dict(torch.load('diffusion_model_final.pth'))

diffusion.eval()

with torch.no_grad():
    samples = diffusion.sample(batch_size=10)

for index, sample in enumerate(samples):
    generated_data = sample.toList()
    events = token_sequence_to_events(generated_data)
    event_sequence_to_midi(events, f'sample_{index}.midi')
