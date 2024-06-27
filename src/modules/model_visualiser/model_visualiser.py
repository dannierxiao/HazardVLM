#%%
import cv2
from matplotlib.colors import Normalize
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import yaml

import torch
from torchvision.transforms.functional import to_tensor
from torchvision.models.feature_extraction import create_feature_extractor
from pytorchvideo.models.hub import x3d_l
from torchvision.models.feature_extraction import get_graph_node_names
from transformers import PreTrainedTokenizerFast
from modules.processing_layers.model_vlm import HazardVLM

class VisualiseActivations:
    """
    Visualise the neuron activations of the encoder during inference using the EigenCAM method.

    EigenCAM chosen as not classification task so no class discrimination required.

    Neuron activates represent output of the neurons in the encoder and are visualised as 2D projections.

    Run evaluation over all the videos in modules/model_visualiser/video/ directory.

    Args:
    - config filename: The configuration file to load for the model
    - video filename: The video filename to load for the model to process
    """

    def __init__(self, config_filename):
        self.rescale_res = 200

        self.save_grid_bool = int(input('Save grid of projections or as individual images? (0/1): '))

        with open(f'config/{config_filename}.yaml', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.model_name = config['model_load_path']

        if not config['model_visualiser_mode']:
            raise ValueError("'model_visualiser_mode' must be enabled in the config file")

        self.output_dir = 'modules/model_visualiser/output'
        if not os.path.exists(self.output_dir): # Ensure the output directory exists
            os.makedirs(self.output_dir)
        
        if not os.path.exists(f'{self.output_dir}/{self.model_name}'): # Ensure the output directory exists
            os.makedirs(f'{self.output_dir}/{self.model_name}')

        self.config = config
        self.tokenizer = PreTrainedTokenizerFast(
                                        tokenizer_file='config/tokenizer/{}.json'.format(self.config['tokenizer']),
                                        pad_token='[PAD]',
                                        sos_token='[SOS]',
                                        eos_token='[EOS]',
                                        sep_token='[SEP]',
                                        cls_token='[CLS]',
                                        mask_token='[MASK]',
                                        unk_token='[UNK]',
                                        )

        self.model = self.load_model()
    
    def main(self):
        for video_filename in os.listdir('modules/model_visualiser/video'):
            if video_filename.endswith('.mp4'):
                self.evaluate(video_filename)
        
        print('Evaluation complete')
        print(f'Output saved to {self.output_dir}/{self.model_name}')


    def evaluate(self, video_filename):
        self.video_name = video_filename.split('.')[0]
        self.video = self.load_video(video_filename)
        self.video = self.video.permute(0, 1, 3, 4, 2) #  -> (B, T, H, W, C) as gets converted in model
        timesteps = self.video.shape[1]
        mask = torch.zeros(1, timesteps, dtype=torch.bool).to('cuda:0') # 0 means no mask
        _, x_act = self.model(x=self.video, mask=mask, captions=None, teacher_forcing_ratio=None) # Extract features from video frames

        for key in x_act:
            x_temp = x_act[key] # Shape after x3d (Batch, Hidden, Timesteps, Feature_Size, Feature_Size)
            x_temp = x_temp.permute(0, 2, 1, 3, 4) #  -> (B, T, C, H, W)
            x_temp = x_temp.squeeze(0) # remove singular batch dimensions
            feature_tensor = x_temp.cpu().detach().numpy()

            projections = []
            for timestep, features in enumerate(feature_tensor):
                projection = self.get_2d_projection(features[np.newaxis, :])  # Your function expects a batch dimension
                projections.append(projection[0])  # Accumulate projections
            if self.save_grid_bool:
                self.save_projections_grid(projections, key)  # Save all projections in a grid
            else:
                for i, projection in enumerate(projections):
                    plt.figure(figsize=(20, 20))
                    plt.imshow(projection, cmap='jet', vmin=0, vmax=1)
                    plt.axis('off')
                    plt.savefig(os.path.join(self.output_dir, self.model_name, f'FRAME_{self.video_name}_{key}_{i}.png'), bbox_inches='tight', dpi=300, transparent=True, pad_inches=0)
                    plt.close()
    
            # self.save_superimposed_grid(projections, video_filename, key)  # Save superimposed grid

    def load_model(self):
        model = HazardVLM(self.config, self.tokenizer)
        model.load_state_dict(torch.load('models/' + self.model_name  +  '.pt'))
        model = model.to('cuda:0')
        model.eval()
        return model

    def load_video(self, video_filename):
        video_filename = f'modules/model_visualiser/video/{video_filename}'

        if not os.path.isfile(video_filename):
            raise FileNotFoundError(f'Video file {video_filename} not found')

        cap = cv2.VideoCapture(video_filename)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (self.rescale_res, self.rescale_res), interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
        finally:
            cap.release()
        
        # Convert frames to a tensor
        video_tensor = torch.stack([to_tensor(frame) for frame in frames])  # Stack frames and convert to tensor
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
        video_tensor = video_tensor.to('cuda:0')
        return video_tensor

    def get_2d_projection(self, activation_batch):
        # From Pytorch Grad-CAM https://github.com/jacobgil/pytorch-grad-cam
        activation_batch[np.isnan(activation_batch)] = 0
        projections = []
        for activations in activation_batch:
            reshaped_activations = (activations).reshape(
                activations.shape[0], -1).transpose()
            # Centering before the SVD, else the image returned is negative
            reshaped_activations = reshaped_activations - \
                reshaped_activations.mean(axis=0)
            U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
            projection = reshaped_activations @ VT[0, :]
            projection = projection.reshape(activations.shape[1:])
            projections.append(projection)
        return np.float32(projections)
    

    def save_projections_grid(self, projections, layer):
        """Normalize and save all projections in a grid to an image file with individual colorbars for the rightmost subplots not on the bottom edge and one large colorbar for the entire grid."""
        num_images = len(projections)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))

        subplot_size = 500  # pixels
        fig_width = cols * subplot_size / 100  # Convert pixels to inches for figure
        fig_height = rows * subplot_size / 100  # Convert pixels to inches for figure
        fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), constrained_layout=False)

        flat_projections = np.concatenate([proj.flatten() for proj in projections])
        min_val = flat_projections.min()
        max_val = flat_projections.max()

        for i, projection in enumerate(projections):
            ax = axs[i // cols, i % cols]
            normalized_projection = (projection - min_val) / (max_val - min_val)
            im = ax.imshow(normalized_projection, cmap='jet', vmin=0, vmax=1)
            ax.axis('off')

        # Create individual colorbars for the rightmost subplots
        for i in range(rows):
            if i * cols + cols - 1 < num_images:  # Check if the rightmost subplot in this row exists
                ax = axs[i, cols - 1]  # Rightmost subplot of each row
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, orientation='vertical')

        for i in range(num_images, rows * cols):
            axs.flat[i].axis('off') # Hide any unused subplots

        plt.savefig(os.path.join(self.output_dir, self.model_name, f'{self.video_name}_{layer}.png'), bbox_inches='tight', dpi=600, transparent=True)
        plt.close()

visualiser = VisualiseActivations(config_filename='model_config_model_vis')
visualiser.main()

# %%
