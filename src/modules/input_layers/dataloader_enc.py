import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from modules.input_layers.data_tools import upsample, get_train_val_test_split

class PreExtractedFeatureDataset(Dataset):
    """
    Dataset loader that loads the pre-extracted encoder features
    """
    def __init__(self, config, tokenizer, split):
        self.config = config
        self.tokenizer = tokenizer

        # Load the dataset
        loaded_data = torch.load('datasets/{}.pt'.format(config['input_filename']))

        if config['upsample_mode']:
            loaded_data = upsample(config, loaded_data)

        self.dict_dataset = get_train_val_test_split(config=config, x=loaded_data['x_features'], captions=loaded_data['x_captions'])

        # Assign the correct split of data to use
        if split == 'train':
            self.features = self.dict_dataset['x_train']
            self.captions = self.dict_dataset['captions_train']
        elif split == 'val':
            self.features = self.dict_dataset['x_val']
            self.captions = self.dict_dataset['captions_val']
        elif split == 'test':
            self.features = self.dict_dataset['x_test']
            self.captions = self.dict_dataset['captions_test']
        else:
            raise ValueError('Invalid split name')

        self.num_samples = len(self.features)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx): # Retrieve the pre-extracted feature and its corresponding caption
        feature = self.features[idx]
        caption = self.captions[idx]
        return feature, caption

    def collate_fn(self, batch):
        video_frames_batch, captions_batch = zip(*batch)
        # print(f"Original lengths: {[len(frames) for frames in video_frames_batch]}")  # Original lengths

        padded_video_frames_batch = pad_sequence(video_frames_batch, batch_first=True)

        video_frames_masks = torch.zeros(padded_video_frames_batch.shape[:2], dtype=torch.bool)
        for i, frames in enumerate(video_frames_batch):
            video_frames_masks[i, :len(frames)] = 1

        # print(f"Mask lengths: {video_frames_masks.sum(dim=1).tolist()}")  # Mask lengths
        # print()
        return padded_video_frames_batch, video_frames_masks, captions_batch
