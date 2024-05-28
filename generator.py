import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D

from data_conversion import token_sequence_to_events, event_sequence_to_midi
from midi_dataset import MIDIDataset
from midi_trainer import MIDITrainer1D

dataset = MIDIDataset(data_root=None, saved_data_path='midi_dataset_full.pth')

model = Unet1D(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length=dataset.max_length,
    timesteps=1000,
    beta_schedule='cosine',
)

trainer = MIDITrainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 60,
    train_lr = 8e-5,
    train_num_steps = 100_000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,
    save_and_sample_every=1000
)

trainer.load(80)
trainer.model.eval()

with torch.no_grad():
    samples = trainer.model.sample(batch_size=20)

for index, sample in enumerate(samples):
    generated_data = sample.tolist()[0]
    events = token_sequence_to_events(generated_data)
    event_sequence_to_midi(events, f'results/samples/sample_{index}.midi')
