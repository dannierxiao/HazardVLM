sweep_name = {
    'name': 'CHPT3 VLM',
}

sweep_config = {
        'method': 'grid', #grid, random, 'bayes' # bayes explained https://wandb.ai/site/articles/bayesian-hyperparameter-optimization-a-primer

        'metric': {
          'name': 'TRAIN LOSS',  #the metric the sweeps are attempting to optimize e.g. loss and goal is to minimise
          'goal': 'minimize'
                  },
        
    
        'parameters':
            {
            'notes': {
            'values': ['']
                  },
            
            'GLOBAL_SEED': {
            'values': [1]
                  },
            
            'dataset': {
            'values': ['DOTA'] # Used for model naming
                  },
            
            'show_haz_actor_bbox': {
            'values': [False] # Whether to show the hazardous actor bounding box in the video
                  },
            
            'image_channels': {
            'values': [3] # Number of channels in the video frames.
                  },
            
            'grayscale': {
            'values': [False] # Whether to convert the video to grayscale
                  },
                        
            'test_split': {
                'values': [0.3] # Split ratio for the test dataset.
            },

            'repeat_run_id': {
            'values': ['A'] # 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J' for multi run model save and comparison
                     },

            'restart_training': {
                'values': [False] # Flag to determine if training should be restarted.
            },

            'upsample_mode': {
                'values': [None] # 'full', 'partial', None # 'full': entire string counts as class, 'partial': first three words, False: no upsample
            },

            'input_filename': {
                'values': ['VLM_DOTA_AS25_MEDI_HCLASS_T10_65_7901', 'VLM_DOTA_AS50_MEDI_HCLASS_T10_65_7901', 'VLM_DOTA_AS75_MEDI_HCLASS_T10_65_7901', 'VLM_DOTA_AS25_RAND_HCLASS_T10_65_7901', 'VLM_DOTA_AS50_RAND_HCLASS_T10_65_7901', 'VLM_DOTA_AS75_RAND_HCLASS_T10_65_7901', 'VLM_DOTA_AS25_UNIF_HCLASS_T10_65_7901', 'VLM_DOTA_AS50_UNIF_HCLASS_T10_65_7901', 'VLM_DOTA_AS75_UNIF_HCLASS_T10_65_7901'] # 'VLM_DOTA_HCLASS_T10_65_7901', 'VLM_DOTA_AS25_MEDI_HCLASS_T10_65_7901', 'VLM_DOTA_AS50_MEDI_HCLASS_T10_65_7901', 'VLM_DOTA_AS75_MEDI_HCLASS_T10_65_7901', 'VLM_DOTA_AS25_RAND_HCLASS_T10_65_7901', 'VLM_DOTA_AS50_RAND_HCLASS_T10_65_7901', 'VLM_DOTA_AS75_RAND_HCLASS_T10_65_7901', 'VLM_DOTA_AS25_UNIF_HCLASS_T10_65_7901', 'VLM_DOTA_AS50_UNIF_HCLASS_T10_65_7901', 'VLM_DOTA_AS75_UNIF_HCLASS_T10_65_7901', 
            },


            # training
            'mixed_precision': {
                'values': [True] # Mixed precision using cuda.amp
            },

            'batch_size': {
                'values': [4] # Number of samples processed before the model is updated.
            },

            'batch_multiplier': {
                'values': [1] # Pseudo batch through accumulation to run on GPUs with lower memory, final batch size = batch * multiplier
            },

            'epochs': {
                'values': [100] # Number of complete passes through the training dataset.
            },

            'lr': {
                'values': [0.001] # Default 0.001 from sweep. 
            },

            'loss_weight_decay': {
                'values': [0.001] # Default 0.001 from sweep. 
            },

            

            'plateau_lr_patience': {
                'values': [10] # Patience for learning rate reduction on plateau.
            },

            'step_lr_multiplier': {
                'values': [0.5] # Factor for step learning rate reduction.
            },

            'optimizer': {
                'values': ['sgd'] # Optimizer type: 'sgd', 'adamW'.
            },

            'early_stop_patience': {
                'values': [20] # Patience before early stopping.
            },

            'loss_func': {
                'values': ['cross_entropy'] # Loss function: 'focal', 'cross_entropy'.
            },

            'hazard_metrics': {
                'values': [True] # Whether to break caption into separate components and score each separately
            },

            'binary_hazard_metrics': {
                'values': [True] # If dataset includes safe class or not
            },

            # Model
            'model': {
                'values': ['vlm'] # For model naming
            },

            'adaptive_frame_sample': {
                'values': [False] # True, False # Whether to use adaptive sampling at runtime. Recommened to preprocess for faster training.
            },

            'adaptive_frame_sample_ratio': {
                'values': [0.5] # Rate of frame sampling.
            },
            'adaptive_frame_sample_mode': {
                'values': ['highest_value'] # 'uniform', 'highest_value', 'random'
            },

            'teacher_forcing_pattern': {
                'values': ['stochastic'] # 'stochastic', 'deterministic'
            },

            'teacher_forcing_decay': {
                'values': [0.03] # Decay per epoch
            },
            'teacher_forcing_decay_epoch': {
                'values': [1] # Decay after this epoch
            },

            'encoder': {
                'values': ['x3d_m'] # Encoder type: 'x3d_l', 'x3d_m'.
            },

            'visual_mlp': {
                'values': [True] # True, False
            },


            'visual_mlp_hidden_layers': {
                'values': [[2048]] # Each hidden layer size [input,out]. Default: [2048]
            },

            'visual_mlp_dropout': {
                'values': [0.75] # MLP dropout rate. Default: 0.75
            },

            'visual_mlp_output_dim': {
                'values': [1200] # Output size from mlp. Needs to be a multiple of decoder heads.
            },



            'decoder_hidden_size': {
                'values': [2048] # Hidden size for the decoder, needs to match visual mlp. # 2048
            },

            'decoder_layers': {
                'values': [12] # Number of layers in the decoder.
            },

            'decoder_heads': {
                'values': [12] # Number of attention heads in the decoder.
            },

            'decoder_dropout': {
                'values': [0.75] # Default 0.75 from sweep. Number of attention heads in the decoder.
            },

            'model_visualiser_mode': {
                'values': [False]
            },
  


            'model_load_path': {
                'values': ['TBD'] # Model load path (without file extension).
            },



            # tokenizer
            'tokenizer': {
                'values': ['tokenizer_dota_A_153_v8'] # Tokenizer name.
            },

            'tokenizer_vocab_size': {
                'values': [153] # Vocabulary size for the tokenizer.
            },

            'max_tok_per_caption': {
                'values': [10] # Maximum tokens per caption.
            },

            'padding_token_idx': {
                'values': [0] # Index for padding token.
            },

            'start_token_idx': {
                'values': [1] # Index for start token.
            },

            'end_token_idx': {
                'values': [2] # Index for end token.
            },
  
                }
        }
