import copy
import os
import time
from datetime import datetime
import statistics

import wandb
import torch
from torch.cuda.amp import autocast

from modules.training_loop.device import device
from modules.training_loop.logging_funct import logging, get_lr
from modules.training_loop.eval_metrics import calculate_metrics, get_caption_components, calc_hazard_detection_metrics, calc_actor_location_metrics, convert_label_to_logits, calc_binary_hazard_detection_metrics
from modules.training_loop.class_mapping import HAZARD_TYPE_DICT, ACTOR_TYPE_DICT, LOCATION_TYPE_DICT
from modules.training_loop.train_tools import clear_memory

def train_model(config, model, train_data_loader, val_data_loader, optimizer, lr_scheduler, loss_func, model_save_name, scaler):
    # Training dataset
    model.train()
    model_weights = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    start_epoch = 0
    no_improv_counter = 0
    patience = config['early_stop_patience']
    total_epochs = config['epochs']

    # Check for existing checkpoint
    latest_checkpoint_path = model_save_name.split('.')[0] +'_checkpoint.pt'
    if config['restart_training'] and os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        logging.info('\n')
        logging.info(f'model restarted from epoch {start_epoch} at checkpoint {latest_checkpoint_path}')
        logging.info('\n')
    else:
        if config['restart_training']:
            raise ValueError(f'Checkpoint file not found for {latest_checkpoint_path}')
    
    # Training details
    pseudo_batch = config['batch_size'] * config['batch_multiplier']  # Pseudo batch through accumulation to run on GPUs with lower memory
    batch_size = config['batch_size']
    max_tok_per_caption = config['max_tok_per_caption']
    tokenizer_vocab_size = config['tokenizer_vocab_size']
    teacher_forcing_ratio = 1

    for epoch in range(start_epoch, total_epochs):
        startTime = time.time() # Time training time
        total_loss = 0

        all_logits_flat = torch.zeros(pseudo_batch * max_tok_per_caption, tokenizer_vocab_size, dtype=torch.float32).to(device)
        all_targets_flat = torch.zeros(pseudo_batch * max_tok_per_caption, dtype=torch.int64).to(device)
        all_masks_flat = torch.zeros(pseudo_batch * max_tok_per_caption, dtype=torch.bool).to(device)
        count = 0

        for idx, (videos, masks, captions) in enumerate(train_data_loader):
            optimizer.zero_grad()  # Clear gradients for the next train step

            # Data preparation
            videos = videos.to(device)
            masks = masks.to(device)
            token_captions_gt = model.tokenizer.batch_encode_plus(captions, max_length=config['max_tok_per_caption'], add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True)
            token_captions_gt = token_captions_gt['input_ids'].to(device)

            # Model outputs
            if config['mixed_precision']:
                with autocast():
                    logits, _ = model(x=videos, mask=masks, captions=token_captions_gt, teacher_forcing_ratio=teacher_forcing_ratio) # Compute the logits using the current decoder input
            else:
                logits, _ = model(x=videos, mask=masks, captions=token_captions_gt, teacher_forcing_ratio=teacher_forcing_ratio) # Compute the logits using the current decoder input

            # Flatten logits for use with Cross-Entropy Loss
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = token_captions_gt.contiguous().view(-1)
            mask = (token_captions_gt != config['padding_token_idx']).contiguous().view(-1) # Create a mask by marking all padding tokens as 0 and everything else as 1

            # Accumulate logits and targets for loss calculation
            temp_start_idx = count*batch_size*max_tok_per_caption
            temp_end_idx = (count+1)*batch_size*max_tok_per_caption

            all_logits_flat[temp_start_idx:temp_end_idx] = logits_flat
            all_targets_flat[temp_start_idx:temp_end_idx] = targets_flat
            all_masks_flat[temp_start_idx:temp_end_idx] = mask
            count += 1

            del videos, masks, token_captions_gt, logits, logits_flat, targets_flat, mask # CLear large temporary variables

            if ((idx + 1) % config['batch_multiplier'] == 0) or ((idx + 1) == len(train_data_loader)):  # Pseudo batch through accumulation to run on GPUs with lower memory
                # Mask out padding tokens
                all_logits_flat = all_logits_flat[all_masks_flat]
                all_targets_flat = all_targets_flat[all_masks_flat]
                
                # Compute the loss over the accumulated samples
                loss = loss_func(all_logits_flat, all_targets_flat) # Compute the loss, only considering non-padding tokens
                total_loss += loss.item()

                # Backpropagation and optimizer step
                if config['mixed_precision']:
                    scaler.scale(loss).backward() # Calls backward() on scaled loss to create scaled gradients
                    # scaler.unscale_(optimizer)  # Unscale the gradients of optimizer's assigned params to allow gradient clipping
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to avoid exploding gradients
                    scaler.step(optimizer) # Update weights
                    scaler.update() # Updates the scale for next iteration
                else:
                    loss.backward() # Perform backpropagation
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to avoid exploding gradients
                    optimizer.step() # Update weights
                
                # Clear the accumulators
                all_logits_flat = torch.zeros(pseudo_batch * max_tok_per_caption, tokenizer_vocab_size, dtype=torch.float32).to(device)
                all_targets_flat = torch.zeros(pseudo_batch * max_tok_per_caption, dtype=torch.int64).to(device)
                all_masks_flat = torch.zeros(pseudo_batch * max_tok_per_caption, dtype=torch.bool).to(device)
                count = 0

            clear_memory()

        avg_train_loss = total_loss / len(train_data_loader)

        if epoch > config['teacher_forcing_decay_epoch']: # start decay after phase 1
            teacher_forcing_ratio = max(0, teacher_forcing_ratio - config['teacher_forcing_decay']) # decay until 0
        
        # Validation set
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():  # Disable gradient calculation
            for idx, (videos, masks, captions) in enumerate(val_data_loader):
                videos = videos.to(device)
                masks = masks.to(device)
                
                logits, _ = model(x=videos, mask=masks, captions=None, teacher_forcing_ratio=None) # Compute the logits using the current decoder input

                # tokenize ground truth captions
                token_captions_gt = model.tokenizer.batch_encode_plus(captions, max_length=config['max_tok_per_caption'], add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True)
                token_captions_gt = token_captions_gt['input_ids'].to(device)

                loss = loss_func(logits.view(-1, config['tokenizer_vocab_size']), token_captions_gt.view(-1))
                total_val_loss += loss.item()

                if (idx + 1) == len(val_data_loader): # Save output of last batch for logging
                    temp_prediction = model.tokenizer.decode(torch.argmax(logits, dim=-1)[0], skip_special_tokens=True) # save for output
                    temp_prediction_ids = torch.argmax(logits, dim=-1)[0].clone().detach().cpu().numpy().tolist() # save for output
                    temp_ground_truth = model.tokenizer.decode(token_captions_gt[0], skip_special_tokens=True) # save for output
                    temp_ground_truth_ids = token_captions_gt[0].clone().detach().cpu().numpy().tolist() # save for output

            avg_val_loss = total_val_loss / len(val_data_loader)
            lr_scheduler.step(avg_val_loss) # decay learning rate by loss

        # Get model updates
        execution_time = (time.time() - startTime)/(60) # Get execution time in minutes
        finish_eta_hours = ((config['epochs']-epoch+1)*execution_time) / 60
        current_time = datetime.now().strftime('%H:%M') # ('%Y-%m-%d %H:%M')
        epoch_log_output = f'{current_time} Epoch {epoch+1}/{total_epochs} | ETA {finish_eta_hours :.2f} hours ({execution_time :.2f} min/epoch) | Train Loss {avg_train_loss :.4f} | Val Loss {avg_val_loss :.4f} |lr {get_lr(optimizer)} | TF Ratio {teacher_forcing_ratio :.2f}'

        wandb.log({
                    'Epoch': epoch,
                    'Epoch Log': epoch_log_output,
                    'Train Loss': avg_train_loss,
                    'Val Loss': avg_val_loss,
                    'Example Output (Val Set)': (temp_prediction, temp_prediction_ids, temp_ground_truth, temp_ground_truth_ids),
                    })

        max_width = 100
        formatted_prediction = (str(temp_prediction_ids) + ' ' + temp_prediction).ljust(max_width)
        formatted_ground_truth = (str(temp_ground_truth_ids) + ' ' + temp_ground_truth).ljust(max_width)
        logging.info(epoch_log_output)
        logging.info(f'-- Model Predict (Val): {formatted_prediction}')
        logging.info(f'-- Ground Truth  (Val): {formatted_ground_truth}')
        logging.info('\n')

        # Save checkpoint at the end of each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_loss': best_loss,
                    }
        torch.save(checkpoint, latest_checkpoint_path)  # Save latest checkpoint

        #save best performance to summarise
        if ((best_loss - avg_val_loss)/best_loss) > 0.001: # if train loss better by at least 0.1%
            best_loss = avg_val_loss
            model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_save_name)
            logging.info('Copied best model weights!')
            no_improv_counter = 0 # Reset counter
        elif epoch == 0: # if first epoch save weights
            best_loss = avg_val_loss
            model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_save_name)
        else:
            no_improv_counter += 1
        
        if (no_improv_counter == patience) or (avg_val_loss < 0.001):
            logging.info(f'Early stopping due to {no_improv_counter} consecutive non-improvements in validation loss')
            break

    os.remove(latest_checkpoint_path) # delete checkpoint file after training has finished
    return model_weights

def eval_model(config, model, test_data_loader, loss_func):
    model.eval()  # Set the model to evaluation mode
    y_pred_dict = {}
    y_pred_dict_ids = {}
    y_pred_dict_logits = {}
    y_gt_dict = {}
    y_gt_dict_ids = {}
    total_eval_loss = 0
    total_pred_time_list = []
    total_avg_pred_time_list = []
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (videos, masks, captions) in enumerate(test_data_loader):
            videos = videos.to(device)
            masks = masks.to(device)

            if torch.cuda.is_available(): # Initialize CUDA timing events
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()  # Start timer
            
            logits, _ = model(x=videos, mask=masks, captions=None, teacher_forcing_ratio=None) # Compute the logits using the current decoder input

            if torch.cuda.is_available(): # Record and compute CUDA event timing
                ender.record()  # End timer
                torch.cuda.synchronize()  # Wait for GPU sync
                pred_time = starter.elapsed_time(ender)  # Calculate inference time
                total_pred_time_list.append(pred_time) # Aggregate total inference time
                total_avg_pred_time_list.append(pred_time / videos.shape[1]) # Aggregate average inference time
                total_samples += 1  # Count the total number of samples processed

            # tokenize ground truth captions
            token_captions_gt = model.tokenizer.batch_encode_plus(captions, max_length=config['max_tok_per_caption'], add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True)
            token_captions_gt = token_captions_gt['input_ids'].to(device)

            loss = loss_func(logits.view(-1, config['tokenizer_vocab_size']), token_captions_gt.view(-1))
            total_eval_loss += loss.item()

            predicted_token_ids = torch.argmax(logits, dim=-1)
            predicted_logits = logits.clone().detach().cpu().numpy().tolist()

            # Decode token IDs to text (captions)
            for idx in range(predicted_token_ids.shape[0]):
                y_pred_dict[f'BATCH{batch_idx}_IDX{idx}'] = [model.tokenizer.decode(predicted_token_ids[idx], skip_special_tokens=True)]
                y_pred_dict_ids[f'BATCH{batch_idx}_IDX{idx}'] = predicted_token_ids[idx].clone().detach().cpu().numpy().tolist()
                y_pred_dict_logits[f'BATCH{batch_idx}_IDX{idx}'] = predicted_logits[idx]
                y_gt_dict[f'BATCH{batch_idx}_IDX{idx}'] = [model.tokenizer.decode(token_captions_gt[idx], skip_special_tokens=True)]
                y_gt_dict_ids[f'BATCH{batch_idx}_IDX{idx}'] = token_captions_gt[idx].clone().detach().cpu().numpy().tolist()

            clear_memory()

        avg_eval_loss = total_eval_loss / len(test_data_loader)
        eval_scores = calculate_metrics(y_gt=y_gt_dict, y_pred=y_pred_dict)

        if config['hazard_metrics']:
            captions_gt = [item[0] for item in y_gt_dict.values()]
            captions_pred = [item[0] for item in y_pred_dict.values()]

            hazard_class_gt_list = []
            actor_class_gt_list = []
            loc_class_gt_list = []

            hazard_class_pred_list = []
            undetected_hazard_pred_list = []
            actor_class_pred_list = []
            loc_class_pred_list = []

            # Iterate through captions and get the class of hazard, actor, location
            for caption in captions_gt:
                hazard_class, actor_class, location_class = get_caption_components(caption)
                hazard_class_gt_list.append(hazard_class)
                actor_class_gt_list.append(actor_class)
                loc_class_gt_list.append(location_class)

            for idx, caption in enumerate(captions_pred):
                hazard_class, actor_class, location_class = get_caption_components(caption)
                hazard_class_pred_list.append(hazard_class)
                actor_class_pred_list.append(actor_class)
                loc_class_pred_list.append(location_class)

                if hazard_class == HAZARD_TYPE_DICT['<unknown>']:
                    undetected_hazard_pred_list.append((captions_gt[idx], caption))
            
            # add undetected index to allow correct auc calc that requires all labels present
            hazard_class_gt_list.append(HAZARD_TYPE_DICT['<unknown>'])
            actor_class_gt_list.append(ACTOR_TYPE_DICT['<unknown>'])
            loc_class_gt_list.append(LOCATION_TYPE_DICT['<unknown>'])

            hazard_class_pred_list.append(HAZARD_TYPE_DICT['<unknown>'])
            actor_class_pred_list.append(ACTOR_TYPE_DICT['<unknown>'])
            loc_class_pred_list.append(LOCATION_TYPE_DICT['<unknown>'])

            hazard_class_pred_logits_norm_list = convert_label_to_logits(hazard_class_pred_list, len(HAZARD_TYPE_DICT))
            actor_class_pred_list_logits_list = convert_label_to_logits(actor_class_pred_list, len(ACTOR_TYPE_DICT))
            loc_class_pred_list_logits_list = convert_label_to_logits(loc_class_pred_list, len(LOCATION_TYPE_DICT))

            eval_scores['hazard_class_gt_list'] = hazard_class_gt_list
            eval_scores['hazard_class_pred_list'] = hazard_class_pred_list
            eval_scores['hazard_class_pred_logits_norm_list'] = hazard_class_pred_logits_norm_list
            eval_scores['hazard_class_unk_percent'] = (hazard_class_pred_list.count(HAZARD_TYPE_DICT['<unknown>']) / len(hazard_class_pred_list)) * 100
            eval_scores['undetected_hazard_pred_list'] = undetected_hazard_pred_list

            eval_scores['actor_class_gt_list'] = actor_class_gt_list
            eval_scores['actor_class_pred_list'] = actor_class_pred_list
            eval_scores['actor_class_pred_logits_list'] = actor_class_pred_list_logits_list
            eval_scores['actor_class_unk_percent'] = (actor_class_pred_list.count(ACTOR_TYPE_DICT['<unknown>']) / len(actor_class_pred_list)) * 100
            
            eval_scores['loc_class_gt_list'] = loc_class_gt_list
            eval_scores['loc_class_pred_list'] = loc_class_pred_list
            eval_scores['loc_class_pred_logits_list'] = loc_class_pred_list_logits_list
            eval_scores['loc_class_unk_percent'] = (loc_class_pred_list.count(LOCATION_TYPE_DICT['<unknown>']) / len(loc_class_pred_list)) * 100

            if config['binary_hazard_metrics']:
                eval_scores = calc_binary_hazard_detection_metrics(eval_scores)

            eval_scores = calc_hazard_detection_metrics(eval_scores)
            eval_scores = calc_actor_location_metrics(eval_scores)

        eval_scores['avg_test_loss'] = avg_eval_loss
        eval_scores['average_pred_time'] = statistics.mean(total_pred_time_list)
        eval_scores['average_avg_pred_time'] = statistics.mean(total_avg_pred_time_list)
        
        if len(total_pred_time_list) > 1:
            eval_scores['std_dev_pred_time'] = statistics.stdev(total_pred_time_list) ## if fails batch size is wrong for test dataset
        else:
            eval_scores['std_dev_pred_time'] = 0

        eval_scores['y_pred_dict'] = y_pred_dict
        eval_scores['y_pred_dict_ids'] = y_pred_dict_ids
        eval_scores['y_pred_dict_logits'] = y_pred_dict_logits
        eval_scores['y_gt_dict'] = y_gt_dict
        eval_scores['y_gt_dict_ids'] = y_gt_dict_ids
        return eval_scores
