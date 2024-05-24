import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D

from analytics import pitch_class_histogram_distances, calculate_pitch_class_distance_histogram
from midi_dataset import MIDIDataset

NUM_EPOCHS = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataset = MIDIDataset(data_root=None, saved_data_path='midi_dataset.pth')
loader = dataset.get_data_loader(batch_size=32)

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

optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

for epoch in range(NUM_EPOCHS):
    real_histograms = []
    generated_histograms = []

    for index, data in enumerate(loader):
        data = data.float()
        loss = diffusion(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate histograms every 1000 iterations
        if index % 1000 == 0:
            generated_sample = diffusion.sample(batch_size=1)[0]
            real_histograms.append(calculate_pitch_class_distance_histogram(generated_sample))
            real_histograms.append(calculate_pitch_class_distance_histogram)

    generated_data = diffusion.sample(batch_size=10)
    distance = pitch_class_histogram_distances(generated_histograms, real_histograms)
    print(f"Epoch {epoch}, Loss: {loss.item()}, Pitch class histogrm distance: {distance}")

    torch.save(diffusion.state_dict(), f'midi_model_{epoch}.pth')