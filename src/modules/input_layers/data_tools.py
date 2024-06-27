import torch
from pathlib import Path
from collections import Counter
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split
import torch
from modules.training_loop.logging_funct import logging
from modules.training_loop.class_mapping import HAZARD_TYPE_DICT, LOCATION_TYPE_DICT, ACTOR_TYPE_DICT

def create_special_tokens_mask(tokenizer, labels):
    """
    Create a mask for special tokens.

    Args:
        tokenizer: Tokenizer object.
        labels (torch.Tensor): Tensor of token ids.

    Returns:
        torch.Tensor: A mask tensor where special tokens are marked as True.
    """
    # Assuming the tokenizer provides the ids of special tokens
    special_token_ids = set(tokenizer.all_special_ids)

    # Create a mask where each special token in labels is marked
    special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
    for token_id in special_token_ids:
        special_tokens_mask |= (labels == token_id)

    return special_tokens_mask


def get_class_indices(captions):
    """
    Temporary indices for stratify, not detection classes
    Returns a list of indices for each temporary class in order of the dataset.
    Temporary class is defined by the first three words of the caption.
    """
    temp_class_idx_list = []

    # Group by the first three words of the caption for class identification
    for caption in captions:
        caption_header = ' '.join(caption.split()[:3])
        if caption_header not in HAZARD_TYPE_DICT:
            raise ValueError(f'Caption header "{caption_header}" not found in HAZARD_TYPE_DICT')
        else:
            temp_class_idx_list.append(HAZARD_TYPE_DICT[caption_header])

    return temp_class_idx_list


def get_train_val_test_split(config, x, captions):
    """
    Function to split the dataset into train, validation, and test sets.
    """

    # Split the data into training and test sets
    test_size = config['test_split']
    list_class_idx = get_class_indices(captions) # Temporary indices for stratify, not detection classes

    # Split the data into training and test sets
    x_train, x_temp, captions_train, captions_temp, _, list_class_idx_temp = train_test_split(
                                                                                            x,
                                                                                            captions,
                                                                                            list_class_idx,
                                                                                            test_size=test_size,
                                                                                            random_state=config['GLOBAL_SEED'],
                                                                                            stratify=list_class_idx)

    x_val, x_test, captions_val, captions_test, = train_test_split(
                                                                    x_temp,
                                                                    captions_temp,
                                                                    test_size=0.5,
                                                                    random_state=config['GLOBAL_SEED'],
                                                                    stratify=list_class_idx_temp)
    
    dict_dataset = {
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'captions_train': captions_train,
        'captions_val': captions_val,
        'captions_test': captions_test,
    }

    return dict_dataset


def upsample(config, dataset):
    """
    Upsamples the minority classes using sklearn's resample function.
    
    :param config: Configuration dictionary with 'upsample_mode' and 'GLOBAL_SEED'.
    :param dataset: Dictionary with 'x_features' tensor and 'x_captions' list.
    :return: Dictionary of the upsampled dataset.
    """
    x_features = dataset['x_features']
    x_captions = dataset['x_captions']

    # Determine the class based on the mode
    if config['upsample_mode'] == 'partial':
        # Group by the first three words of the caption for class identification
        class_to_indices = {}
        for i, caption in enumerate(x_captions):
            class_label = ' '.join(caption.split()[:3])
            if class_label not in class_to_indices:
                class_to_indices[class_label] = []
            class_to_indices[class_label].append(i)
        classes = [' '.join(caption.split()[:3]) for caption in x_captions]
    else:
        # Use the full caption for class identification
        class_to_indices = {caption: [i] for i, caption in enumerate(x_captions)}
        classes = x_captions

    # Find the class with the maximum samples for upsampling target
    max_samples = max(len(indices) for indices in class_to_indices.values())

    # Upsample each class to have the same number of samples as the max_samples
    resampled_features = []
    resampled_captions = []

    for class_label, indices in class_to_indices.items():
        class_features = [x_features[i] for i in indices]
        class_captions = [x_captions[i] for i in indices]

        if len(class_captions) != max_samples:
            # Upsample the current class samples
            upsampled_features, upsampled_captions = resample(
                class_features,
                class_captions,
                replace=True,
                n_samples=max_samples - len(class_captions), # Upsample only the amount needed
                random_state=config['GLOBAL_SEED']  # For reproducibility
            )

            # Append the original and upsampled samples to the main list
            resampled_features.extend(class_features + upsampled_features)
            resampled_captions.extend(class_captions + upsampled_captions)
        else:
            resampled_features.extend(class_features)
            resampled_captions.extend(class_captions)

    # Convert features back to tensor
    resampled_features_tensor = torch.stack(resampled_features, dim=0)

    upsampled_dataset = {'x_features': resampled_features_tensor,
                         'x_captions': resampled_captions}

    logging.info('Upsampling Information:')
    logging.info(f'- Unique Classes: {len(set(classes))}, Biggest Class: {max_samples}')
    logging.info(f'- Dataset size before: {len(x_captions)}, after: {len(resampled_captions)}')
    logging.info(f'- {Counter(classes)}')
    logging.info('\n')

    return upsampled_dataset


def upsample_file_list(config, video_files, captions):
    """
    Upsamples the minority classes using sklearn's resample function.
    
    :param config: Configuration dictionary with 'upsample_mode' and 'GLOBAL_SEED'.
    :param dataset: Dictionary with 'x_features' tensor and 'captions' list.
    :return: Dictionary of the upsampled dataset.
    """
    # Determine the class based on the mode
    if config['upsample_mode'] == 'partial':
        # Group by the first three words of the caption for class identification
        class_to_indices = {}
        for i, caption in enumerate(captions):
            class_label = ' '.join(caption.split()[:3])
            if class_label not in class_to_indices:
                class_to_indices[class_label] = []
            class_to_indices[class_label].append(i)
        classes = [' '.join(caption.split()[:3]) for caption in captions]
    else:
        # Use the full caption for class identification
        class_to_indices = {caption: [i] for i, caption in enumerate(captions)}
        classes = captions

    # Find the class with the maximum samples for upsampling target
    max_samples = max(len(indices) for indices in class_to_indices.values())

    # Upsample each class to have the same number of samples as the max_samples
    resampled_video_files = []
    resampled_captions = []

    for class_label, indices in class_to_indices.items():
        class_videos = [video_files[i] for i in indices]
        class_captions = [captions[i] for i in indices]

        if len(class_captions) != max_samples:
            # Upsample the current class samples
            upsampled_features, upsampled_captions = resample(
                class_videos,
                class_captions,
                replace=True,
                n_samples=max_samples - len(class_captions), # Upsample only the amount needed
                random_state=config['GLOBAL_SEED']  # For reproducibility
            )

            # Append the original and upsampled samples to the main list
            resampled_video_files.extend(class_videos + upsampled_features)
            resampled_captions.extend(class_captions + upsampled_captions)
        else:
            resampled_video_files.extend(class_videos)
            resampled_captions.extend(class_captions)


    logging.info('Upsampling Information:')
    logging.info(f'- Unique Classes: {len(set(classes))}, Biggest Class: {max_samples}')
    logging.info(f'- Dataset size before: {len(captions)}, after: {len(resampled_captions)}')
    logging.info(f'- {Counter(classes)}')
    logging.info('\n')

    return resampled_video_files, resampled_captions
