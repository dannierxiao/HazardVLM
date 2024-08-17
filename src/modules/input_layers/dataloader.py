#%%
import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from modules.training_loop.logging_funct import logging
from modules.input_layers.data_tools import upsample_file_list, get_train_val_test_split
from modules.input_layers.adaptive_sampling_tools import get_sample_frames, get_sample_frames_threading

class HazardVideoDataset(Dataset):
    def __init__(self, config, split, end_frames_removed=0.0):
        self.config = config
        self.end_frames_removed = end_frames_removed

        # Load data from JSON file
        with open('datasets/{}.json'.format(config['input_filename']), 'r') as file:
            data = json.load(file)

        video_files = []
        for item in data:
            if os.name == 'nt':  # Windows
                normalized_path = item['filename'].replace('/', '\\')
            else:  # Unix/Linux
                normalized_path = item['filename'].replace('\\', '/')
            
            video_files.append(normalized_path)

        captions = [item['metadata']['caption'] for item in data]

        if self.config['upsample_mode']:
            video_files, captions = upsample_file_list(config, video_files, captions)

        if self.config['show_haz_actor_bbox']:
            self.video_data = [item['metadata'] for item in data]

        # Split the data into training and test sets
        self.dict_dataset = get_train_val_test_split(config=config, x=video_files, captions=captions)

        # Assign the correct split of data to use
        if split == 'train':
            self.video_files = self.dict_dataset['x_train']
            self.captions = self.dict_dataset['captions_train']
        elif split == 'val':
            self.video_files = self.dict_dataset['x_val']
            self.captions = self.dict_dataset['captions_val']
        elif split == 'test':
            self.video_files = self.dict_dataset['x_test']
            self.captions = self.dict_dataset['captions_test']
        elif split == 'all':
            self.video_files = self.dict_dataset['x_train'] + self.dict_dataset['x_val'] + self.dict_dataset['x_test']
            self.captions = self.dict_dataset['captions_train'] + self.dict_dataset['captions_val'] + self.dict_dataset['captions_test']
        else:
            raise ValueError('Invalid split name')

        self.num_samples = len(self.video_files)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.config['show_haz_actor_bbox']:
            video_data = self.video_data[idx]
        else:
            video_data = None

        # Load video
        video_path = self.video_files[idx]

        if self.config['adaptive_frame_sample']:
            video_frames = self.load_video_with_adaptive_sampling(video_path, video_data)
        else:
            video_frames = self.load_video(video_path, video_data)
        caption_tensor = self.captions[idx]
        return video_frames, caption_tensor

    def load_video(self, video_path, video_data=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Video path is not accessible or the video cannot be opened."

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_load = total_frames - int(total_frames * self.end_frames_removed)
        frames = []
        frame_idx = 0

        while frame_idx < frames_to_load:
            ret, frame = cap.read()
            if not ret:
                break

            if self.config['show_haz_actor_bbox']:
                frame_annotations = video_data['haz_actor_bbox'][frame_idx]
                if frame_annotations: # If frame has annotation, draw bounding boxes
                    for dict in frame_annotations:
                        actor_bbox = dict['bbox']
                        pt1 = (int(actor_bbox[0]), int(actor_bbox[1]))
                        pt2 = (int(actor_bbox[2]), int(actor_bbox[3]))
                        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 5)

            if self.config['grayscale']:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_tensor = torch.from_numpy(frame).float() / 255.0 # convert to float and normalize to [0, 1]
                frame_tensor = frame_tensor.unsqueeze(-1) # add single channel dimension needed for later permute
            else:
                frame_tensor = torch.from_numpy(frame).float() / 255.0 # convert to float and normalize to [0, 1]

            frames.append(frame_tensor)
            frame_idx += 1
        cap.release()
        return torch.stack(frames)

    def collate_fn(self, batch):
        video_frames_batch, captions_batch = zip(*batch)
        padded_video_frames_batch = pad_sequence(video_frames_batch, batch_first=True)
        padded_captions_batch = captions_batch

        video_frames_masks = torch.zeros(padded_video_frames_batch.shape[:2], dtype=torch.bool)
        for i, frames in enumerate(video_frames_batch):
            video_frames_masks[i, :len(frames)] = 1

        return padded_video_frames_batch, video_frames_masks, padded_captions_batch


    def load_video_with_adaptive_sampling(self, video_path, video_data=None):
        """
        Load a video and adaptively sample frames based on optical flow.

        Parameters:
        video_path (str): The file path of the video to be processed.

        Returns:
        list: A sorted list of frame indices that have been adaptively sampled from the video.
        """
        sampling_ratio = self.config['adaptive_frame_sample_ratio'] # The ratio of the total frames to be sampled. For example, 0.1 means sampling 10% of the frames.
        if sampling_ratio == 0:
            raise ValueError("Sampling ratio must be > 0")

        if self.config['adaptive_frame_sample_mode'] and self.config['adaptive_frame_sample_threading']:
            sampled_frames = get_sample_frames_threading(self.config, video_path)
        else:
            sampled_frames = get_sample_frames(self.config, video_path, video_data)
        
        # Reload the video to process and extract the sampled frames
        cap = cv2.VideoCapture(video_path)
        processed_frames = []
        for frame_idx in sampled_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            if self.config['grayscale']:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_tensor = torch.from_numpy(frame).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(-1)
            else:
                frame_tensor = torch.from_numpy(frame).float() / 255.0
            processed_frames.append(frame_tensor)
        
        cap.release()
        return torch.stack(processed_frames)