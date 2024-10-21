import os
import glob
import torch
import numpy as np


class VideoLatentDataset(torch.utils.data.Dataset):
    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.video_lists = [d for d in os.listdir(self.data_path)]
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames

    def __len__(self):
        return len(self.video_lists)

    def load_latents(self, latent_path):
        latents = torch.load(latent_path)  # Load the entire latent tensor
        return latents

    def __getitem__(self, index):
        sample = self.video_lists[index]
        latent_path = os.path.join(self.data_path, sample)

        # Load latents and calculate total frames
        latents = self.load_latents(latent_path)  # Latent shape: [num_frames, channels, height, width]
        total_frames = latents.shape[0]

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)

        # Select latent frames
        latent_window = latents[frame_indice]

        return {'video_latent': latent_window, 'video_name': 1}


class FramesLatentDataset(torch.utils.data.Dataset):
    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.video_lists = [d for d in os.listdir(self.data_path)]
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames

    def __len__(self):
        return len(self.video_lists)

    def load_latents(self, folder_path, selected_indices):
        latents = []
        for idx in selected_indices:
            latent_path = os.path.join(folder_path, f"{idx+1:06d}.npz")  # Change .pt to .npz
            latent_data = np.load(latent_path)['latent']  # Load latent data from .npz file
            latent = torch.tensor(latent_data)  # Convert numpy array to PyTorch tensor
            latents.append(latent)
        return torch.stack(latents)  # Return a tensor of stacked latents

    def __getitem__(self, index):
        video_folder_path = os.path.join(self.data_path, self.video_lists[index])
        frame_paths = sorted(glob.glob(os.path.join(video_folder_path, '*.jpg')))  # Assuming frames are saved as .jpg
        total_frames = len(frame_paths)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len

        # Get the specific frame indices to load
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)

        # Load the corresponding frames
        latents = self.load_latents(video_folder_path, frame_indices)
        return {'video_latent': latents, 'video_name': 1}


if __name__ == '__main__':
    pass