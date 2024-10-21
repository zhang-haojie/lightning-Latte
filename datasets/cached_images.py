import os
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers.models import AutoencoderKL
import video_transforms


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
        self.video_lists = [d for d in os.listdir(self.data_path)]
        self.transform = transform
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)

        # Load the VAE model
        self.vae = AutoencoderKL.from_pretrained(vae_model_path)
        self.vae.requires_grad_(False)
        self.vae.to("cuda")

    def __len__(self):
        return len(self.video_lists)

    def load_images(self, folder_path):
        # List jpg images and keep original names for matching
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        image_files.sort()  # Sort by name to keep the order
        return image_files  # Return sorted image file names

    def __getitem__(self, index):
        sample_folder = self.video_lists[index]
        folder_path = os.path.join(self.data_path, sample_folder)
        image_files = self.load_images(folder_path)  # Load sorted image file names

        transformed_images = []
        valid_image_files = []  # To keep track of image files whose latents don't exist
        for image_file in image_files:
            # Check if the latent file already exists
            latent_file_name = image_file.replace('.jpg', '.npz')
            latent_path = os.path.join(self.target_dir, sample_folder, latent_file_name)

            if os.path.exists(latent_path):
                # If latent file exists, skip this image file
                continue

            # If latent file does not exist, process the image
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            transformed_images.append(image)
            valid_image_files.append(image_file)  # Track this image file for latent saving later

        # If no valid images, return or handle as necessary
        if len(transformed_images) == 0:
            return

        images = torch.stack(transformed_images)  # (N, C, H, W)
        images = images.to("cuda")

        all_latents = []
        batch_size = 64
        for i in range(0, images.shape[0], batch_size):
            batch_images = images[i:i + batch_size]
            with torch.no_grad():
                latents = self.vae.encode(batch_images).latent_dist.sample().mul_(0.18215)
            all_latents.append(latents.cpu())
            torch.cuda.empty_cache()

        latents = torch.cat(all_latents, dim=0)

        # Save latents for valid image files only
        for image_file, latent in zip(valid_image_files, latents):
            latent_file_name = image_file.replace('.jpg', '.npz')
            latent_path = os.path.join(self.target_dir, sample_folder, latent_file_name)
            os.makedirs(os.path.dirname(latent_path), exist_ok=True)

            latent_numpy = latent.cpu().numpy()
            np.savez_compressed(latent_path, latent=latent_numpy)

        return


if __name__ == "__main__":
    data_path = "/path/to/datasets/preprocess_ffs/train/images"
    vae_model_path = "/path/to/pretrained/Latte/vae"
    target_dir = "/path/to/datasets/cached_images"
    os.makedirs(target_dir, exist_ok=True)

    image_size = 256    # 256/512
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = CachedFaceForensicsDataset(data_path, vae_model_path, target_dir, transform)

    for i in tqdm(range(len(dataset)), desc="Processing folders"):
        dataset[i]