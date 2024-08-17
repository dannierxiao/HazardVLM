# Standard library imports
import os
import random
from sys import platform
import time

# Related third-party imports
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
import yaml
import wandb
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer

# Local application/library specific imports
from modules.training_loop.device import device
from modules.training_loop.logging_funct import logging
from modules.training_loop.model_params_count import count_parameters
from modules.training_loop.eval_metrics import set_debug_apis, log_model_metrics, log_vlm_metrics, log_hazard_detection_metrics, log_binary_hazard_detection_metrics, log_actor_location_metrics
from modules.training_loop.train_tools import set_visible_device
from modules.input_layers.dataloader import HazardVideoDataset as CustomDataLoader

with open('config/init_config.yaml', encoding='utf-8') as f:
    INIT_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

set_visible_device(device=INIT_CONFIG['target_gpu'])

logging.info('VLM MODE')
time.sleep(1) # wait 1 seconds to allow user to read
from modules.processing_layers.model_vlm import HazardVLM
from modules.training_loop.train_eval_vlm import train_model, eval_model
from config.model_config_vlm_sweep import sweep_config, sweep_name
from modules.training_loop.eval_metrics import calc_flops_vlm as calc_flops

with open('config/model_config_vlm.yaml', encoding='utf-8') as f:
    MODEL_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

if MODEL_CONFIG['encoder'] == 'pre_extracted':
    from modules.input_layers.dataloader_enc import PreExtractedFeatureDataset as CustomDataLoader

set_debug_apis(state=False) # disable debugging apis for faster training for final model
if INIT_CONFIG['wandb_bool'] == 0:
    logging.info('*** TEST MODE')
    WANDB_MODE = 'disabled'
    RUN_NAME = 'prototyping_run'
    WANDB_ACC = INIT_CONFIG['wandb_account_name']

    logging.info('\n')
    logging.info('WANDB LOGGING DISABLED FOR DEBUGGING --------------------')
    logging.info('\n')
else:
    WANDB_MODE = 'online'
    RUN_NAME = INIT_CONFIG['run_name']
    WANDB_ACC = INIT_CONFIG['wandb_account_name']

logging.info('-'*50)
logging.info('\n')


# Unique model ID
model_unique_no = random.randint(10000,99999) # gen unique model no. before global seed is set

# Config files
os.environ['WANDB_AGENT_MAX_INITIAL_FAILURES'] = '1' # terminate sweep upon error
os.environ['WANDB_AGENT_FLAPPING_MAX_FAILURES'] = '1' # terminate sweep upon error
os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true" # terminate sweep upon error

GLOBAL_SEED = MODEL_CONFIG['GLOBAL_SEED'] # seed for reproducibility
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)

if INIT_CONFIG['cudnn_deterministic']:
    torch.backends.cudnn.deterministic = True # set true for cuDNN to make results reproducible


def model_init(model_config=None): # none by default if left empty
    # Initialize a new wandb run
    if INIT_CONFIG['single_run_bool'] == 0:
        project_name = None
    else:
        project_name = model_config['project_name']

    with wandb.init(project=project_name, entity=WANDB_ACC, config=model_config, mode=WANDB_MODE, name=RUN_NAME):
        model_config = wandb.config # this config will be set by Sweep Controller

        # Init tokenizer
        if model_config['tokenizer'] == 'gpt2':
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token_id = 0
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
            tokenizer.mask_token_id = 3
            tokenizer.unk_token_id = 4
        else:
            tokenizer = PreTrainedTokenizerFast(
                                        tokenizer_file='config/tokenizer/{}.json'.format(model_config['tokenizer']),
                                        pad_token='[PAD]',
                                        sos_token='[SOS]',
                                        eos_token='[EOS]',
                                        sep_token='[SEP]',
                                        cls_token='[CLS]',
                                        mask_token='[MASK]',
                                        unk_token='[UNK]',
                                        )
    
        # Check tokenizer vocab size
        if tokenizer.vocab_size != model_config['tokenizer_vocab_size']:
            raise ValueError('Tokenizer vocab size {} does not match config file {}'.format(tokenizer.vocab_size , model_config['tokenizer_vocab_size']))

        model = HazardVLM(model_config, tokenizer)
        model = model.to(device)
        
        # Initialize the datasets
        train_dataset = CustomDataLoader(model_config, split='train')
        val_dataset = CustomDataLoader(model_config, split='val')
        test_dataset = CustomDataLoader(model_config, split='test')
        test_dataset_frames_removed = CustomDataLoader(model_config, split='test', end_frames_removed=0.5)

        # Initialize the DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True, collate_fn=train_dataset.collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=model_config['batch_size'], shuffle=False, collate_fn=test_dataset.collate_fn, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn, drop_last=True)
        test_loader_frames_removed = DataLoader(test_dataset_frames_removed, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn, drop_last=True)

        # FLOPs calculation 
        logging.info("FLOPs Calculation ---------------------")
        logging.info('\n')
        eval_scores = {}
        eval_scores = calc_flops(model, test_loader, eval_scores)
        macs = eval_scores['macs']
        flops = eval_scores['flops']
        logging.info('\n')

        # Compile model
        if INIT_CONFIG['compile']:
            if platform == 'linux' or platform == 'linux2':
                logging.info(f'*** Torch compile enabled ***')
                model = torch.compile(model)
            else:
                logging.info(f'*** Torch compile not supported on {platform} at time of publication ***')

        # Evaluation or transfer learning on pretrained model
        if INIT_CONFIG['proc_mode'] in [2,3]:
            model_save_name = model_config['model_load_path']
            model.load_state_dict(torch.load('models/' + model_save_name  +  '.pt'))
            logging.info('Model load success')
            logging.info('\n')

        model_params = count_parameters(model)
        logging.info('\n')

        logging.info('Train/Val/Test Split Information:')
        logging.info(f'- Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
        logging.info('\n')

        if model_config['loss_func'] == 'cross_entropy':
            loss_func = nn.CrossEntropyLoss(reduction='mean').to(device)  # Compute the loss, only considering non-padding tokens

        # Define the optimizer
        if model_config['optimizer']=='sgd':
            optimizer = torch.optim.SGD(model.parameters(),lr=model_config['lr'], weight_decay=model_config['loss_weight_decay'])
        elif model_config['optimizer']=='adamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=model_config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=model_config['loss_weight_decay'])

        if model_config['mixed_precision']:
            scaler = GradScaler()
            logging.info('*** Mixed point precision on ***')
            logging.info('\n')
        else:
            scaler = None

        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=model_config['step_lr_multiplier'], patience=model_config['plateau_lr_patience'],verbose=1) #factor=0.5, patience=5,verbose=1
        os.makedirs('./models', exist_ok=True)

        if INIT_CONFIG['proc_mode'] == 1:
            if model_config['restart_training']: # if training interrupted and needs reload from checkpoint
                model_save_name = 'models/' + model_config['model_load_path'] + '.pt'
                logging.info(f'Model reloaded {model_save_name}')
            else:
                model_save_name = 'models/' + str(model_config['model']) + '_' + str(model_config['dataset']) + '_' + str(model_unique_no) + '_' + str(model_config['repeat_run_id'])  + '.pt'
                logging.info(f'Model save name {model_save_name}')

            logging.info("Training ---------------------")
            trained_model_weights = train_model(
                                                config=model_config,
                                                model=model,
                                                train_data_loader=train_loader,
                                                val_data_loader=val_loader,
                                                optimizer=optimizer,
                                                lr_scheduler=lr_scheduler,
                                                loss_func=loss_func,
                                                model_save_name=model_save_name,
                                                scaler=scaler
                                                )

            model.load_state_dict(trained_model_weights)

        logging.info('\n')
        logging.info("Evaluation 1. full video ---------------------")
        logging.info('\n')
        eval_scores = eval_model(
                                config=model_config,
                                model=model,
                                test_data_loader=test_loader,
                                loss_func=loss_func
                                )

        logging.info("Evaluation 2. video with end frames removed ---------------------")
        eval_scores_frames_removed = eval_model(
                                                config=model_config,
                                                model=model,
                                                test_data_loader=test_loader_frames_removed,
                                                loss_func=loss_func
                                                )
        logging.info('\n')

        eval_scores['macs'] = macs
        eval_scores['flops'] = flops
        eval_scores['model_save_name'] = model_save_name
        eval_scores['model_params'] = model_params

        log_model_metrics(eval_scores)

        if INIT_CONFIG['mode'] == 'vlm':
            log_vlm_metrics(eval_scores=eval_scores, end_frames_removed=False)
            log_vlm_metrics(eval_scores=eval_scores_frames_removed, end_frames_removed=True)

        if MODEL_CONFIG['binary_hazard_metrics']:
            log_binary_hazard_detection_metrics(eval_scores=eval_scores, end_frames_removed=False)
            log_binary_hazard_detection_metrics(eval_scores=eval_scores_frames_removed, end_frames_removed=True)

        if MODEL_CONFIG['hazard_metrics']:
            log_hazard_detection_metrics(eval_scores=eval_scores, end_frames_removed=False)
            log_hazard_detection_metrics(eval_scores=eval_scores_frames_removed, end_frames_removed=True)

            if INIT_CONFIG['mode'] == 'vlm':
                log_actor_location_metrics(eval_scores=eval_scores, end_frames_removed=False)
                log_actor_location_metrics(eval_scores=eval_scores_frames_removed, end_frames_removed=True)

        logging.info('\n')
        logging.info('Evaluation metrics logged')
        logging.info('\n')
        logging.info('Finished ---------------------')

        # if INIT_CONFIG['proc_mode'] == 1 and eval_scores['avg_test_loss'] > 1:
        #     os.remove(model_save_name)
        #     logging.info('Only archiving model files with test loss 1. Deleting model due to test loss of {:.2f}.'.format(eval_scores['avg_test_loss']))

######################################################################################
# Main
######################################################################################

if __name__ == '__main__':
    single_run_bool = INIT_CONFIG['single_run_bool']
    torch.backends.cudnn.benchmark = True # cuDNN Autotuner to auto test and select most efficient convolution algos to use
    torch.backends.cudnn.enabled = True

    if single_run_bool == 0:
        sweep_id = wandb.sweep(sweep_config, project=sweep_name['name']) # init sweep
        wandb.agent(sweep_id, model_init, count=100)
        logging.info('\n')
        logging.info('------------------- Finished! ---------------------')
    else: #in training mode
        CONFIG = MODEL_CONFIG
        model_init(CONFIG)
        logging.info('\n')
        logging.info('------------------- Finished! ---------------------')


