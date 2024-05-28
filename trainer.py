import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D

from midi_dataset import MIDIDataset
from midi_trainer import MIDITrainer1D

NUM_EPOCHS = 10

dataset = MIDIDataset(data_root=None, saved_data_path='midi_dataset_small.pth')
loader = dataset.get_data_loader(batch_size=8)

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
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,
    save_and_sample_every=1
)
trainer.train()

torch.save(diffusion.state_dict(), f'midi_model.pth')