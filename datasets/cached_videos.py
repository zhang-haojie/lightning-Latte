import os
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers.models import AutoencoderKL
import video_transforms


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
            # Filelist.append( filename)
    return Filelist


class CachedFaceForensicsDataset(torch.utils.data.Dataset):
    """Load and cache the entire FaceForensics video dataset.
    
    Args:
        configs: configuration object with data_path and other parameters.
        vae_model_path (str): Path to the pretrained VAE model.
        target_dir (str): Directory where the cached latent vectors will be saved.
        transform (callable): Transformation to apply on video frames.
    """
    
    def __init__(self, data_path, vae_model_path, target_dir, transform=None):
        self.data_path = data_path
        self.video_lists = get_filelist(self.data_path)  # Retrieve the list of videos
        self.transform = transform
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)

        # Load the VAE model
        self.vae = AutoencoderKL.from_pretrained(vae_model_path)
        self.vae.requires_grad_(False)
        self.vae.to("cuda")

    def __len__(self):
        return len(self.video_lists)

    def __getitem__(self, index):
        # Load all video frames
        path = self.video_lists[index]
        video_name = os.path.splitext(os.path.basename(path))[0]
        latent_path = os.path.join(self.target_dir, f"{video_name}.pt")
        # If latent vector is already cached, skip processing
        if os.path.exists(latent_path):
            return

        vframes, _, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')

        # Apply transformations to the video frames
        if self.transform:
            video = self.transform(vframes)  # T C H W format

        video = video.to("cuda")
        all_latents = []

        batch_size = 64
        for i in range(0, video.shape[0], batch_size):
            batch_video = video[i:i + batch_size]
            with torch.no_grad():
                latents = self.vae.encode(batch_video).latent_dist.sample().mul_(0.18215)
            all_latents.append(latents.cpu())
            torch.cuda.empty_cache()

        latents = torch.cat(all_latents, dim=0)
        torch.save(latents, latent_path)

        return


if __name__ == "__main__":
    data_path = "/path/to/datasets/preprocess_ffs/train/videos"
    vae_model_path = "/path/to/pretrained/Latte/vae"
    target_dir = "/path/to/datasets/cached_images"

    image_size = 256
    # Define the transformations
    transform_ffs = transforms.Compose([
        video_transforms.ToTensorVideo(), # TCHW
        # video_transforms.RandomHorizontalFlipVideo(),
        video_transforms.UCFCenterCropVideo(image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = CachedFaceForensicsDataset(data_path, vae_model_path, target_dir, transform=transform_ffs)

    for i in tqdm(range(len(dataset))):
        dataset[i]
