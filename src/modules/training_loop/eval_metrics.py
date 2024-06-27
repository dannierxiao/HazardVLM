import numpy as np
import math
import platform
import wandb
import torch
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from evalcap.cider.pyciderevalcap.ciderD.ciderD import CiderD
from thop import profile

from modules.training_loop.logging_funct import logging
from modules.training_loop.device import device
from modules.training_loop.class_mapping import HAZARD_TYPE_DICT, LOCATION_TYPE_DICT, ACTOR_TYPE_DICT

def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state) # creates an annotated timeline for your run that can be visualized by NVIDIA Visual Profiler (NVP)
    torch.autograd.set_detect_anomaly(mode=state)

def calculate_metrics(y_gt, y_pred):
    """Based on pycocoevalcap eval.py

    Takes in pretokenized ground truth and predicted captions and returns
    dictionary of scores for each metric.
    """
    eval_scores = {}

    if platform.system() == 'Windows':
        scorers = [
                    (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
                    (Rouge(),"ROUGE_L"),
                    (Cider(), "Cider")
                ]
        eval_scores['METEOR'] = 'PROTO DATASET'
    else: 
        scorers = [
                    (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
                    (Meteor(),"METEOR"),
                    (Rouge(),"ROUGE_L"),
                    (Cider(), "Cider"),
                    # (CiderD(), "CiderD")
                ]
    for scorer, method in scorers:
        # try:
        score, scores = scorer.compute_score(y_gt, y_pred)
        if type(score)==list:
            for m,s in zip(method,score):
                eval_scores[m] = s
        else:
            eval_scores[method] = score

    return eval_scores

def format_scientific(value):
    """
    Adjusts a float value slightly to ensure its scientific notation is YAML 1.1 compatible
    when serialized, without converting it to a string.
    """
    if isinstance(value, float) and 0 < abs(value) < 1e-4:
        # Determine if the value is in the range that causes formatting issues
        log_value = math.log10(abs(value))
        adjusted_value = round(value, int(-log_value) + 1)
        return adjusted_value
    return value


def log_model_metrics(eval_scores):
    wandb.log({
               'Model Name' : eval_scores['model_save_name'],
               'Model Parms' : format_scientific(eval_scores['model_params']),
               'Model Parms (SCI)' : f"{eval_scores['model_params']:.3e}".replace("+", ""),
               'Model FLOPS': format_scientific(eval_scores['flops']),
               'Model FLOPS (SCI)': f"{eval_scores['flops']:.3e}".replace("+", ""),
               'Model MACS': format_scientific(eval_scores['macs']),
               'Model MACS (SCI)': f"{eval_scores['macs']:.3e}".replace("+", ""),

               'Pred Time AVG (/VIDEO IN MS)': eval_scores['average_pred_time'],
               'Pred Time AVG (/FRAME IN MS)': eval_scores['average_avg_pred_time'],
               'Pred Time STD DEV (/VIDEO IN MS)': eval_scores['std_dev_pred_time'],
            })

def log_vlm_metrics(eval_scores, end_frames_removed):
    df_model_pred = pd.DataFrame({
    'filename': list(eval_scores['y_pred_dict'].keys()),
    'ground_truth': [item[0] for item in eval_scores['y_gt_dict'].values()],
    'ground_truth_ids': [str(item) for item in eval_scores['y_gt_dict_ids'].values()],
    'prediction': [item[0] for item in eval_scores['y_pred_dict'].values()],
    'prediction_ids': [str(item) for item in eval_scores['y_pred_dict_ids'].values()],
    'prediction_logits': [str(item) for item in eval_scores['y_pred_dict_logits'].values()],
    })

    wandb_model_pred_table = wandb.Table(dataframe=df_model_pred)

    if end_frames_removed:
        wandb.log({
               'Pred Time AVG (/VIDEO IN MS) (½ FRAMES)': eval_scores['average_pred_time'],
               'Pred Time AVG (/FRAMES IN MS) (½ FRAMES)': eval_scores['average_avg_pred_time'],
               'Pred Time STD DEV (/VIDEO IN MS) (½ FRAMES)': eval_scores['std_dev_pred_time'],

               'Test Loss (½ FRAMES)': eval_scores['avg_test_loss'],
               'Bleu_1 (½ FRAMES)': eval_scores['Bleu_1'],
               'Bleu_2 (½ FRAMES)': eval_scores['Bleu_2'],
               'Bleu_3 (½ FRAMES)': eval_scores['Bleu_3'],
               'Bleu_4 (½ FRAMES)': eval_scores['Bleu_4'],
               'METEOR (½ FRAMES)': eval_scores['METEOR'],
               'ROUGE_L (½ FRAMES)': eval_scores['ROUGE_L'], 
               'Cider (½ FRAMES)': eval_scores['Cider'],
               'Model Pred Table (½ FRAMES)': wandb_model_pred_table # workaround for wandb table logging issue
                })
    else:
        wandb.log({
                'Test Loss': eval_scores['avg_test_loss'],
                'Bleu_1': eval_scores['Bleu_1'],
                'Bleu_2': eval_scores['Bleu_2'],
                'Bleu_3': eval_scores['Bleu_3'],
                'Bleu_4': eval_scores['Bleu_4'],
                'METEOR': eval_scores['METEOR'],
                'ROUGE_L': eval_scores['ROUGE_L'], 
                'Cider': eval_scores['Cider'],
                'Model Pred Table': wandb_model_pred_table # workaround for wandb table logging issue
                    })
    

def calc_flops_vlm(model, dataloader, eval_scores):
    """
    Function to calculate the FLOPS of a model using the pytorch-OpCounter library.
    """
    (video_frames_0, batch_video_masks_0, _) = next(iter(dataloader))
    inputs = (video_frames_0, batch_video_masks_0, None, 0) # captions=None, tf_ratio=None
    model.eval()
    macs, params = profile(model, inputs=(video_frames_0.to(device), batch_video_masks_0.to(device), None, 0))
    flops = macs * 2 # FLOPS = MACs * 2
    eval_scores['macs'] = macs
    eval_scores['flops'] = flops
    return eval_scores

def calc_flops_benchmark(model, dataloader, eval_scores):
    """
    Function to calculate the FLOPS of a model using the pytorch-OpCounter library.
    """
    (video_frames_0, _, _) = next(iter(dataloader))
    video_frames_0 = video_frames_0.permute(0, 4, 1, 2, 3) # [B, T, H, W, C] -> [B, C, T, H, W]
    model.eval()

    macs, params = profile(model, inputs=(video_frames_0.to(device), ))
    flops = macs * 2 # FLOPS = MACs * 2
    eval_scores['macs'] = macs
    eval_scores['flops'] = flops
    return eval_scores


def get_caption_components(caption):
    """
    Function to extract the hazard type, relation type, and actor type from the caption.
    """

    hazard_idx = HAZARD_TYPE_DICT['<unknown>']
    actor_idx = ACTOR_TYPE_DICT['<unknown>']
    location_idx = LOCATION_TYPE_DICT['<unknown>']

    # Split the string into words
    words = caption.split()

    # Extract the first three words to identify the event type
    event_seq_str = ' '.join(words[:3])
    if event_seq_str in HAZARD_TYPE_DICT:
        hazard_idx = HAZARD_TYPE_DICT[event_seq_str]

    if hazard_idx == 0:
        actor_idx, location_idx = 0, 0
        return hazard_idx, actor_idx, location_idx

    # Join last three words to identify the location
    loc_seq_str = ' '.join(words[-2:])
    if loc_seq_str in LOCATION_TYPE_DICT:
        location_idx = LOCATION_TYPE_DICT[loc_seq_str]

    # Find the actor by splitting into individual words and searching for the first match
    for word in words:
        if word in ACTOR_TYPE_DICT:
            actor_idx = ACTOR_TYPE_DICT[word]
            break

    return hazard_idx, actor_idx, location_idx


def convert_label_to_logits(predicted_labels, num_classes):
    """
    Convert predicted labels to pseudo logits (num samples, num classes) for auc calculation.
    """
    # Initialize the 2D array with zeros
    one_hot_encoded = np.zeros((len(predicted_labels), num_classes))

    for i, label in enumerate(predicted_labels):
        if label < num_classes: # Set to 1 if the label is within the valid range
            one_hot_encoded[i, label] = 1
        else: # Incorrect predictions
            one_hot_encoded[i] = np.ones(num_classes) / num_classes # Distribute the score equally to show uncertainty, as scores need to sum to 1 for auc
    return one_hot_encoded

def calc_binary_hazard_detection_metrics(eval_scores):
    hazard_class_gt_list = eval_scores['hazard_class_gt_list']
    hazard_class_pred_list = eval_scores['hazard_class_pred_list']

    # Binary hazard classification
    binary_hazard_class_gt_list = [1 if item != 0 else 0 for item in hazard_class_gt_list]
    binary_hazard_class_pred_list = [1 if item != 0 else 0 for item in hazard_class_pred_list]
    binary_conf_matrix_haz = confusion_matrix(binary_hazard_class_gt_list, binary_hazard_class_pred_list)
    fpr = binary_conf_matrix_haz[0][1] / (binary_conf_matrix_haz[0][1] + binary_conf_matrix_haz[0][0]) #FPR = FP/(FP+TN)
    fnr = binary_conf_matrix_haz[1][0] / (binary_conf_matrix_haz[1][0] + binary_conf_matrix_haz[1][1]) #FNR = FN/(FN+TP)
    eval_scores['accuracy (hazard) (binary class)'] = accuracy_score(binary_hazard_class_gt_list, binary_hazard_class_pred_list)
    eval_scores['precision (hazard) (binary class)'] = precision_score(binary_hazard_class_gt_list, binary_hazard_class_pred_list, average='binary', zero_division=0)
    eval_scores['recall (hazard) (binary class)'] = recall_score(binary_hazard_class_gt_list, binary_hazard_class_pred_list, average='binary', zero_division=0)
    eval_scores['f1 (hazard) (binary class)'] = f1_score(binary_hazard_class_gt_list, binary_hazard_class_pred_list, average='binary', zero_division=0)
    eval_scores['roc_auc (hazard) (binary class)'] = roc_auc_score(binary_hazard_class_gt_list, binary_hazard_class_pred_list)
    eval_scores['fpr (hazard) (binary class)'] = fpr
    eval_scores['fnr (hazard) (binary class)'] = fnr
    eval_scores['conf_matrix (hazard) (binary class)'] = binary_conf_matrix_haz

    return eval_scores

def calc_hazard_detection_metrics(eval_scores):
    hazard_class_gt_list = eval_scores['hazard_class_gt_list']
    hazard_class_pred_list = eval_scores['hazard_class_pred_list']
    hazard_class_pred_logits_norm_list = eval_scores['hazard_class_pred_logits_norm_list'] # logits normalised between 0 and 1, and sum to 1 for auc calculation

    calc_haz_auc = True if len(set(hazard_class_gt_list)) == len(HAZARD_TYPE_DICT) else False # If test dataset, set to False

    # Multiclass hazard classification
    hazard_class_labels = list(HAZARD_TYPE_DICT.values())

    eval_scores['conf_matrix (hazard)'] = confusion_matrix(hazard_class_gt_list, hazard_class_pred_list)
    eval_scores['precision (hazard) (by class)'] = precision_score(hazard_class_gt_list, hazard_class_pred_list, average=None, zero_division=0)
    eval_scores['recall (hazard) (by class)'] = recall_score(hazard_class_gt_list, hazard_class_pred_list, average=None, zero_division=0)
    eval_scores['f1 (hazard) (by class)'] = f1_score(hazard_class_gt_list, hazard_class_pred_list, average=None, zero_division=0)
    
    eval_scores['precision (hazard) (macro)'] = precision_score(hazard_class_gt_list, hazard_class_pred_list, average='macro', zero_division=0)
    eval_scores['recall (hazard) (macro)'] = recall_score(hazard_class_gt_list, hazard_class_pred_list, average='macro', zero_division=0)
    eval_scores['f1 (hazard) (macro)'] = f1_score(hazard_class_gt_list, hazard_class_pred_list, average='macro', zero_division=0)   

    eval_scores['precision (hazard) (micro)'] = precision_score(hazard_class_gt_list, hazard_class_pred_list, average='micro', zero_division=0)
    eval_scores['recall (hazard) (micro)'] = recall_score(hazard_class_gt_list, hazard_class_pred_list, average='micro', zero_division=0)
    eval_scores['f1 (hazard) (micro)'] = f1_score(hazard_class_gt_list, hazard_class_pred_list, average='micro', zero_division=0)

    # AUC calculation
    if calc_haz_auc:
        eval_scores['roc_auc (hazard) (by class)'] = roc_auc_score(hazard_class_gt_list, hazard_class_pred_logits_norm_list, average=None, multi_class='ovr', labels=hazard_class_labels)
        eval_scores['roc_auc (hazard) (macro)'] = roc_auc_score(hazard_class_gt_list, hazard_class_pred_logits_norm_list, average='macro', multi_class='ovr', labels=hazard_class_labels)
        eval_scores['roc_auc (hazard) (micro)'] = roc_auc_score(hazard_class_gt_list, hazard_class_pred_logits_norm_list, average='micro', multi_class='ovr', labels=hazard_class_labels)
    else:
        eval_scores['roc_auc (hazard) (by class)'] = 'N/A SMALL TEST DATASET WITHOUT ALL CLASSES'
        eval_scores['roc_auc (hazard) (macro)'] = 'N/A SMALL TEST DATASET WITHOUT ALL CLASSES'
        eval_scores['roc_auc (hazard) (micro)'] = 'N/A SMALL TEST DATASET WITHOUT ALL CLASSES'

    return eval_scores

def calc_actor_location_metrics(eval_scores):
    actor_class_gt_list = eval_scores['actor_class_gt_list']
    actor_class_pred_list = eval_scores['actor_class_pred_list']
    actor_class_pred_logits_list = eval_scores['actor_class_pred_logits_list']

    loc_class_gt_list = eval_scores['loc_class_gt_list']
    loc_class_pred_list = eval_scores['loc_class_pred_list']
    loc_class_pred_logits_list = eval_scores['loc_class_pred_logits_list']

    calc_actor_auc = True if len(set(actor_class_gt_list)) == len(ACTOR_TYPE_DICT) else False # If test dataset, set to False
    calc_loc_auc = True if len(set(loc_class_gt_list)) == len(LOCATION_TYPE_DICT) else False # If test dataset, set to False

    # Multiclass actor classification
    actor_class_labels = list(ACTOR_TYPE_DICT.values())
    eval_scores['conf_matrix (actor)'] = confusion_matrix(actor_class_gt_list, actor_class_pred_list)
    eval_scores['accuracy (actor)'] = accuracy_score(actor_class_gt_list, actor_class_pred_list)
    eval_scores['precision (actor) (by class)'] = precision_score(actor_class_gt_list, actor_class_pred_list, average=None, zero_division=0)
    eval_scores['recall (actor) (by class)'] = recall_score(actor_class_gt_list, actor_class_pred_list, average=None, zero_division=0)
    eval_scores['f1 (actor) (by class)'] = f1_score(actor_class_gt_list, actor_class_pred_list, average=None, zero_division=0)

    eval_scores['precision (actor) (macro)'] = precision_score(actor_class_gt_list, actor_class_pred_list, average='macro', zero_division=0)
    eval_scores['recall (actor) (macro)'] = recall_score(actor_class_gt_list, actor_class_pred_list, average='macro', zero_division=0)
    eval_scores['f1 (actor) (macro)'] = f1_score(actor_class_gt_list, actor_class_pred_list, average='macro', zero_division=0)
    
    eval_scores['precision (actor) (micro)'] = precision_score(actor_class_gt_list, actor_class_pred_list, average='micro', zero_division=0)
    eval_scores['recall (actor) (micro)'] = recall_score(actor_class_gt_list, actor_class_pred_list, average='micro', zero_division=0)
    eval_scores['f1 (actor) (micro)'] = f1_score(actor_class_gt_list, actor_class_pred_list, average='micro', zero_division=0)

    # Multiclass location classification
    loc_class_labels = list(LOCATION_TYPE_DICT.values())
    eval_scores['conf_matrix (location)'] = confusion_matrix(loc_class_gt_list, loc_class_pred_list)
    eval_scores['accuracy (location)'] = accuracy_score(loc_class_gt_list, loc_class_pred_list)
    eval_scores['precision (location) (by class)'] = precision_score(loc_class_gt_list, loc_class_pred_list, average=None, zero_division=0)
    eval_scores['recall (location) (by class)'] = recall_score(loc_class_gt_list, loc_class_pred_list, average=None, zero_division=0)
    eval_scores['f1 (location) (by class)'] = f1_score(loc_class_gt_list, loc_class_pred_list, average=None, zero_division=0)

    eval_scores['precision (location) (macro)'] = precision_score(loc_class_gt_list, loc_class_pred_list, average='macro', zero_division=0)
    eval_scores['recall (location) (macro)'] = recall_score(loc_class_gt_list, loc_class_pred_list, average='macro', zero_division=0)
    eval_scores['f1 (location) (macro)'] = f1_score(loc_class_gt_list, loc_class_pred_list, average='macro', zero_division=0)

    eval_scores['precision (location) (micro)'] = precision_score(loc_class_gt_list, loc_class_pred_list, average='micro', zero_division=0)
    eval_scores['recall (location) (micro)'] = recall_score(loc_class_gt_list, loc_class_pred_list, average='micro', zero_division=0)
    eval_scores['f1 (location) (micro)'] = f1_score(loc_class_gt_list, loc_class_pred_list, average='micro', zero_division=0)

    if calc_actor_auc:
        eval_scores['roc_auc (actor) (by class)'] = roc_auc_score(actor_class_gt_list, actor_class_pred_logits_list, average=None,multi_class='ovr', labels=actor_class_labels)
        eval_scores['roc_auc (actor) (macro)'] = roc_auc_score(actor_class_gt_list, actor_class_pred_logits_list, average='macro',multi_class='ovr', labels=actor_class_labels)
        eval_scores['roc_auc (actor) (micro)'] = roc_auc_score(actor_class_gt_list, actor_class_pred_logits_list, average='micro',multi_class='ovr', labels=actor_class_labels)
    else:
        eval_scores['roc_auc (actor) (by class)'] = 'N/A SMALL TEST DATASET WITHOUT ALL CLASSES'
        eval_scores['roc_auc (actor) (macro)'] = 'N/A SMALL TEST DATASET WITHOUT ALL CLASSES'
        eval_scores['roc_auc (actor) (micro)'] = 'N/A SMALL TEST DATASET WITHOUT ALL CLASSES'

    if calc_loc_auc:
        eval_scores['roc_auc (location) (by class)'] = roc_auc_score(loc_class_gt_list, loc_class_pred_logits_list, average=None, multi_class='ovr', labels=loc_class_labels)
        eval_scores['roc_auc (location) (macro)'] = roc_auc_score(loc_class_gt_list, loc_class_pred_logits_list, average='macro', multi_class='ovr', labels=loc_class_labels)
        eval_scores['roc_auc (location) (micro)'] = roc_auc_score(loc_class_gt_list, loc_class_pred_logits_list, average='micro', multi_class='ovr', labels=loc_class_labels)
    else:
        eval_scores['roc_auc (location) (by class)'] = 'N/A SMALL TEST DATASET WITHOUT ALL CLASSES'
        eval_scores['roc_auc (location) (macro)'] = 'N/A SMALL TEST DATASET WITHOUT ALL CLASSES'
        eval_scores['roc_auc (location) (micro)'] = 'N/A SMALL TEST DATASET WITHOUT ALL CLASSES'
    
    return eval_scores


def log_binary_hazard_detection_metrics(eval_scores, end_frames_removed):
    hazard_log_dict = {
                        # Binary class hazard detection metrics
                        'CONF_MATRIX (HAZARD) (BINARY)': eval_scores['conf_matrix (hazard) (binary class)'],
                        'ACC (HAZARD) (BINARY)': eval_scores['accuracy (hazard) (binary class)'],
                        'PREC (HAZARD) (BINARY)': eval_scores['precision (hazard) (binary class)'],
                        'RECALL (HAZARD) (BINARY)': eval_scores['recall (hazard) (binary class)'],
                        'F1 (HAZARD) (BINARY)': eval_scores['f1 (hazard) (binary class)'],
                        'AUC (HAZARD) (BINARY)': eval_scores['roc_auc (hazard) (binary class)'],
                        'FPR (HAZARD) (BINARY)': eval_scores['fpr (hazard) (binary class)'],
                        'FNR (HAZARD) (BINARY)': eval_scores['fnr (hazard) (binary class)'],
                        }

    if end_frames_removed:
        hazard_log_dict = {f"{key} (½ FRAMES)": value for key, value in hazard_log_dict.items()} # Add '(½ FRAMES)' to the key
    
    wandb.log(hazard_log_dict)

def log_hazard_detection_metrics(eval_scores, end_frames_removed):
    hazard_log_dict = {
                        # Multiclass hazard detection metrics
                        'UNDETECTED PRED LIST (HAZARD) (GT, PRED)': eval_scores['undetected_hazard_pred_list'],
                        'UNDETECTED (%) (HAZARD)': eval_scores['hazard_class_unk_percent'],
                        'CONF_MATRIX (HAZARD)': eval_scores['conf_matrix (hazard)'],
                        'PREC (HAZARD) (BY CLASS)': eval_scores['precision (hazard) (by class)'],
                        'RECALL (HAZARD) (BY CLASS)': eval_scores['recall (hazard) (by class)'],
                        'F1 (HAZARD) (BY CLASS)': eval_scores['f1 (hazard) (by class)'],
                        'AUC (HAZARD) (BY CLASS)': eval_scores['roc_auc (hazard) (by class)'],

                        'PREC (HAZARD) (MACRO)': eval_scores['precision (hazard) (macro)'],
                        'RECALL (HAZARD) (MACRO)': eval_scores['recall (hazard) (macro)'],
                        'F1 (HAZARD) (MACRO)': eval_scores['f1 (hazard) (macro)'],
                        'AUC (HAZARD) (MACRO)': eval_scores['roc_auc (hazard) (macro)'],

                        'PREC (HAZARD) (MICRO)': eval_scores['precision (hazard) (micro)'],
                        'RECALL (HAZARD) (MICRO)': eval_scores['recall (hazard) (micro)'],
                        'F1 (HAZARD) (MICRO)': eval_scores['f1 (hazard) (micro)'],
                        'AUC (HAZARD) (MICRO)': eval_scores['roc_auc (hazard) (micro)'],        
                        }

    if end_frames_removed:
        hazard_log_dict = {f"{key} (½ FRAMES)": value for key, value in hazard_log_dict.items()} # Add '(½ FRAMES)' to the key
    
    wandb.log(hazard_log_dict)

def log_actor_location_metrics(eval_scores, end_frames_removed):
    actor_location_log_dict = {
                                # Multiclass actor detection metrics
                                'UNDETECTED (%) (ACTOR)': eval_scores['actor_class_unk_percent'],
                                'CONF_MATRIX (ACTOR)': eval_scores['conf_matrix (actor)'],
                                'ACC (ACTOR)': eval_scores['accuracy (actor)'],
                                'PREC (ACTOR) (BY CLASS)': eval_scores['precision (actor) (by class)'],
                                'RECALL (ACTOR) (BY CLASS)': eval_scores['recall (actor) (by class)'],
                                'F1 (ACTOR) (BY CLASS)': eval_scores['f1 (actor) (by class)'],
                                'AUC (ACTOR) (BY CLASS)': eval_scores['roc_auc (actor) (by class)'],

                                'PREC (ACTOR) (MACRO)': eval_scores['precision (actor) (macro)'],
                                'RECALL (ACTOR) (MACRO)': eval_scores['recall (actor) (macro)'],
                                'F1 (ACTOR) (MACRO)': eval_scores['f1 (actor) (macro)'],
                                'AUC (ACTOR) (MACRO)': eval_scores['roc_auc (actor) (macro)'],

                                'PREC (ACTOR) (MICRO)': eval_scores['precision (actor) (micro)'],
                                'RECALL (ACTOR) (MICRO)': eval_scores['recall (actor) (micro)'],
                                'F1 (ACTOR) (MICRO)': eval_scores['f1 (actor) (micro)'],
                                'AUC (ACTOR) (MICRO)': eval_scores['roc_auc (actor) (micro)'],

                                # Multiclass location detection metrics
                                'UNDETECTED (%) (LOCATION)': eval_scores['loc_class_unk_percent'],
                                'CONF_MATRIX (LOCATION)': eval_scores['conf_matrix (location)'],
                                'ACC (LOCATION)': eval_scores['accuracy (location)'],
                                'PREC (LOCATION) (BY CLASS)': eval_scores['precision (location) (by class)'],
                                'RECALL (LOCATION) (BY CLASS)': eval_scores['recall (location) (by class)'],
                                'F1 (LOCATION) (BY CLASS)': eval_scores['f1 (location) (by class)'],
                                'AUC (LOCATION) (BY CLASS)': eval_scores['roc_auc (location) (by class)'],

                                'PREC (LOCATION) (MACRO)': eval_scores['precision (location) (macro)'],
                                'RECALL (LOCATION) (MACRO)': eval_scores['recall (location) (macro)'],
                                'F1 (LOCATION) (MACRO)': eval_scores['f1 (location) (macro)'],
                                'AUC (LOCATION) (MACRO)': eval_scores['roc_auc (location) (macro)'],

                                'PREC (LOCATION) (MICRO)': eval_scores['precision (location) (micro)'],
                                'RECALL (LOCATION) (MICRO)': eval_scores['recall (location) (micro)'],
                                'F1 (LOCATION) (MICRO)': eval_scores['f1 (location) (micro)'],
                                'AUC (LOCATION) (MICRO)': eval_scores['roc_auc (location) (micro)'],
                                }
    
    if end_frames_removed:
        actor_location_log_dict = {f"{key} (½ FRAMES)": value for key, value in actor_location_log_dict.items()} # Add '(½ FRAMES)' to the key
    wandb.log(actor_location_log_dict)
