import cv2
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from concurrent.futures import ThreadPoolExecutor

def calculate_optical_flow_cuda(prev_frame, current_frame):
    """
    Calculate the optical flow between two consecutive frames using the Farneback method.

    Parameters:
    prev_frame (ndarray): The previous frame in the video sequence, in grayscale.
    current_frame (ndarray): The current frame in the video sequence, in grayscale.

    Returns:
    float: The average magnitude of the optical flow vectors across the frame, representing the average motion.

    The Farneback method calculates dense optical flow, providing a vector field where each vector shows 
    the motion of points from the first frame to the second. Key parameters for the Farneback method include:
    """
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_gray_gpu = cv2.cuda_GpuMat()
    prev_gray_gpu.upload(prev_gray) # Transfer to GPU
    current_gray_gpu = cv2.cuda_GpuMat()
    current_gray_gpu.upload(current_gray) # Transfer to GPU
    
    # Calculate the dense optical flow using Farneback method
    optical_flow = cv2.cuda_FarnebackOpticalFlow.create(
                                                        numLevels=3,   # pyramid levels
                                                        pyrScale=0.5,  # scale between levels in the pyramid
                                                        winSize=15,    # averaging window size
                                                        numIters=3,    # iterations at each pyramid level
                                                        polyN=5,       # size of the pixel neighborhood
                                                        polySigma=1.2, # standard deviation of the Gaussian
                                                        flags=0        # flags
                                                    )

    flow = cv2.cuda_GpuMat() # Create a GPU matrix to store the flow
    flow_gpu = optical_flow.calc(prev_gray_gpu, current_gray_gpu, flow, None) # Compute optical flow
    flow = flow_gpu.download() # Transfer to CPU
    
    # Compute the magnitude and angle of the flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Return the average magnitude of optical flow
    return np.median(magnitude)


def calculate_optical_flow(prev_frame, current_frame):
    """
    Calculate the optical flow between two consecutive frames using the Farneback method.

    Parameters:
    prev_frame (ndarray): The previous frame in the video sequence, in grayscale.
    current_frame (ndarray): The current frame in the video sequence, in grayscale.

    Returns:
    float: The average magnitude of the optical flow vectors across the frame, representing the average motion.

    The Farneback method calculates dense optical flow, providing a vector field where each vector shows 
    the motion of points from the first frame to the second.
    """
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  
    numLevels = 3   # pyramid levels
    pyrScale = 0.5  # scale between levels in the pyramid
    winSize = 15    # averaging window size
    numIters = 3    # iterations at each pyramid level
    polyN = 5       # size of the pixel neighborhood
    polySigma = 1.2 # standard deviation of the Gaussian
    flags = 0       # flags

    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # Compute the magnitude and angle of the flow vectors
    return np.median(magnitude)

def get_sample_frames(config, video_path, video_data=None):
    """
    If the adaptive frame sampling mode is set to 'uniform', the function will sample frames uniformly across the video.
    If the adaptive frame sampling mode is set to 'highest_value', the function will sample frames with the highest optical flow magnitude.
    
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Error: Video path is not accessible or the video cannot be opened {video_path}."
    
    sampling_ratio = config['adaptive_frame_sample_ratio']
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_samples = max(1, math.floor(total_frames * sampling_ratio))
    sampled_frames = []
    prev_frame = None

    if config['adaptive_frame_sample_mode'] == 'uniform':
        chunk_size = max(1, total_frames // num_samples)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"Error: Video path is not accessible or the video cannot be opened {video_path}."
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_load = total_frames - int(total_frames * config['end_frames_removed'])
        
        num_samples = max(1, math.floor(frames_to_load * sampling_ratio))
        chunk_size = max(1, frames_to_load // num_samples)
        
        sampled_frames = []
        prev_frame = None

        for i in range(0, frames_to_load, chunk_size): # i is the starting frame of the chunk
            max_flow_magnitude = -1
            selected_frame_idx = None
            for j in range(chunk_size): # j is the frame index within the chunk
                if i + j >= frames_to_load: # If the frame index exceeds the total frames, as i is the chunk index + j is how many frames along in the chunk
                    break
                ret, current_frame = cap.read()
                if not ret:
                    break
                if config['show_haz_actor_bbox'] and video_data:
                    frame_annotations = video_data['haz_actor_bbox'].get(i + j, [])
                    if frame_annotations:
                        for annotation in frame_annotations:
                            bbox = annotation['bbox']
                            cv2.rectangle(current_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 5)
                if prev_frame is not None:
                    flow_magnitude = self.calculate_optical_flow(prev_frame, current_frame)
                    if flow_magnitude > max_flow_magnitude:
                        max_flow_magnitude = flow_magnitude
                        selected_frame_idx = i + j
                prev_frame = current_frame

            if selected_frame_idx is not None:
                sampled_frames.append(selected_frame_idx)

    elif config['adaptive_frame_sample_mode'] == 'highest_value':
        flow_magnitudes = []
        for i in range(total_frames):
            ret, current_frame = cap.read()
            if not ret:
                break
            if config['show_haz_actor_bbox'] and video_data:
                frame_annotations = video_data['haz_actor_bbox'].get(i, [])
                if frame_annotations:
                    for annotation in frame_annotations:
                        bbox = annotation['bbox']
                        cv2.rectangle(current_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 5)
            # Calculate optical flow if there's a previous frame to compare with
            if prev_frame is not None:
                flow_magnitude = calculate_optical_flow(prev_frame, current_frame)
                flow_magnitudes.append((flow_magnitude, i)) # Store the flow magnitude along with the frame index
            prev_frame = current_frame
        
        flow_magnitudes.sort(key=lambda x: x[0], reverse=True) # Sort high to low
        sampled_frames = [idx for _, idx in flow_magnitudes[:num_samples]] # Select the top n frames based on num_samples
    
    elif config['adaptive_frame_sample_mode'] == 'random':
        frame_indices = list(range(total_frames))
        sampled_frames = random.sample(frame_indices, num_samples)

    else:
        raise ValueError('Invalid adaptive frame sampling mode: {}'.format(config['adaptive_frame_sample_mode']))
    
    cap.release()
    sampled_frames.sort()
    return sampled_frames

def visualize_optical_flow(video_path):
    """
    Helper function to visualize the optical flow between frames in a video given video path.
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    overlay_visualizations = []
    magnitude_images = []

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break
        
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(prev_frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        gray_rgb = cv2.cvtColor(next_gray, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(gray_rgb, 0.5, flow_rgb, 0.5, 0)
        overlay_visualizations.append(overlay)

        normalized_magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        magnitude_images.append(normalized_magnitude)

        prev_gray = next_gray
    
    cap.release()

    grid_size = math.ceil(math.sqrt(len(overlay_visualizations)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))  # Adjust figsize as needed

    for i in range(grid_size):
        for j in range(grid_size):
            ax = axs[i, j]
            index = i * grid_size + j
            if index < len(overlay_visualizations):
                im = ax.imshow(cv2.cvtColor(overlay_visualizations[index], cv2.COLOR_BGR2RGB))
                ax.axis('off')
                
                # Only add color bars to the rightmost axis in each row
                if j == grid_size - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap='jet'), cax=cax, orientation='vertical')
            else:
                ax.axis('off')

    plt.tight_layout()
    plt.savefig('scene_frames_grid_optical_flow.png')

def visualise_processed_frames(processed_frames):
    """
    Helper function to visualize the processed frames in a grid layout.
    """
    # Determine the grid size - creating a square grid that can fit all frames
    grid_size = math.ceil(math.sqrt(len(processed_frames)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))  # Reduced figure size
    sampled_frame_indices = [i for i in range(len(processed_frames))]
    for i in range(grid_size):
        for j in range(grid_size):
            ax = axs[i, j]
            index = i * grid_size + j
            if index < len(processed_frames):
                frame = processed_frames[index].numpy()
                if frame.shape[0] == 1:  # If grayscale, remove the channel dimension
                    frame = np.squeeze(frame, axis=0)
                elif frame.shape[0] == 3:  # If RGB, transpose to get (H, W, C)
                    frame = np.transpose(frame, (1, 2, 0))

                ax.imshow(frame, cmap='gray' if frame.shape[0] == 1 else None)
                ax.set_title(f"Frame: {sampled_frame_indices[index]}", fontsize=8)
                ax.axis('off')
            else:
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'scene_frames_grid.png')
    plt.close()


def process_chunk(chunk_start, chunk_size, frames):
    max_flow_magnitude = -1
    selected_frame_idx = None
    prev_frame = None

    for j in range(chunk_size):
        frame_idx = chunk_start + j
        if frame_idx >= len(frames):
            break

        current_frame = frames[frame_idx]

        if prev_frame is not None:
            flow_magnitude = calculate_optical_flow(prev_frame, current_frame)
            if flow_magnitude > max_flow_magnitude:
                max_flow_magnitude = flow_magnitude
                selected_frame_idx = frame_idx

        prev_frame = current_frame

    return selected_frame_idx


def get_sample_frames_threading(config, video_path):
    num_threads = os.cpu_count()
    
    # Load video to frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Error: Video path is not accessible or the video cannot be opened {video_path}."

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()  # Release the VideoCapture object since we no longer need it

    sampling_ratio = config['adaptive_frame_sample_ratio']
    if sampling_ratio == 0:
        raise ValueError("Sampling ratio must be > 0")

    total_frames = len(frames)
    num_samples = max(1, math.floor(total_frames * sampling_ratio))
    chunk_size = max(1, total_frames // num_samples)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(0, total_frames, chunk_size):
            futures.append(executor.submit(process_chunk, i, chunk_size, frames))
        
        sampled_frames = [f.result() for f in futures if f.result() is not None]

    sampled_frames.sort()
    return sampled_frames
