import os
import glob
import json
import torch
import decord
import torchvision

import numpy as np


from PIL import Image
from einops import rearrange
from typing import Dict, List, Tuple

class_labels_map = None
cls_sample_cnt = None

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def numpy2tensor(x):
    return torch.from_numpy(x)


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
            # Filelist.append( filename)
    return Filelist


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(num_class, anno_pth='./k400_classmap.json'):
    global class_labels_map, cls_sample_cnt
    
    if class_labels_map is not None:
        return class_labels_map, cls_sample_cnt
    else:
        cls_sample_cnt = {}
        class_labels_map = load_annotation_data(anno_pth)
        for cls in class_labels_map:
            cls_sample_cnt[cls] = 0
        return class_labels_map, cls_sample_cnt


def load_annotations(ann_file, num_class, num_samples_per_cls):
    dataset = []
    class_to_idx, cls_sample_cnt = get_class_labels(num_class)
    with open(ann_file, 'r') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            sample = {}
            idx = 0
            # idx for frame_dir
            frame_dir = line_split[idx]
            sample['video'] = frame_dir
            idx += 1
                                
            # idx for label[s]
            label = [x for x in line_split[idx:]]
            assert label, f'missing label in line: {line}'
            assert len(label) == 1
            class_name = label[0]
            class_index = int(class_to_idx[class_name])
            
            # choose a class subset of whole dataset
            if class_index < num_class:
                sample['label'] = class_index
                if cls_sample_cnt[class_name] < num_samples_per_cls:
                    dataset.append(sample)
                    cls_sample_cnt[class_name]+=1

    return dataset


class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1, **kwargs):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.kwargs = kwargs
        
    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str


class FaceForensics(torch.utils.data.Dataset):
    """Load the FaceForensics video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.video_lists = get_filelist(configs.data_path)
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()

    def __getitem__(self, index):
        path = self.video_lists[index]
        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)
        
        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
        video = vframes[frame_indice]
        # videotransformer data proprecess
        video = self.transform(video) # T C H W
        return {'video': video, 'video_name': 1}

    def __len__(self):
        return len(self.video_lists)


class FaceForensicsFrames(torch.utils.data.Dataset):
    """Load the FaceForensics video frames from extracted image folders
    
    Args:
        target_video_len (int): the number of video frames to be loaded.
        align_transform (callable): Align different videos to a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.video_lists = [d for d in os.listdir(self.data_path)]
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames

    def load_images(self, folder_path, selected_indices):
        images = []
        for idx in selected_indices:
            img_path = os.path.join(folder_path, f"{idx+1:06d}.jpg")
            img = Image.open(img_path).convert("RGB")
            if img is None:
                raise FileNotFoundError(f"Image {img_path} not found.")
            images.append(img)
        return np.array(images)

    def __getitem__(self, index):
        # Get the folder path for the current video
        video_folder_path = os.path.join(self.data_path, self.video_lists[index])
        frame_paths = sorted(glob.glob(os.path.join(video_folder_path, '*.jpg')))  # Assuming frames are saved as .jpg
        total_frames = len(frame_paths)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len

        # Get the specific frame indices to load
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)

        # Load the corresponding frames
        video = self.load_images(video_folder_path, frame_indices)
        video = torch.tensor(np.transpose(video, (0, 3, 1, 2)))

        # Apply the transform to the entire sequence of frames
        if self.transform:
            video = self.transform(video)

        return {'video': video, 'video_name': 1}

    def __len__(self):
        return len(self.video_lists)


if __name__ == '__main__':
    pass