import math
import random
import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights  # Example 3D ResNet
from torchvision.models.feature_extraction import create_feature_extractor
from pytorchvideo.models.hub import x3d_l, x3d_m

from modules.training_loop.device import device
from modules.training_loop.logging_funct import logging
from modules.processing_layers.visual_mlp import VideoFeatureMLP

class HazardVLM(nn.Module):
    def __init__(self, config, tokenizer):
        """
        Hazard aware video lanaguage model.
        Utilising an encoder to extract the visual characterisitcs of a video and a transformer for generating text sequences.
            
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_length = config['max_tok_per_caption']
        self.start_token_idx = config['start_token_idx']
        self.end_token_idx = config['end_token_idx']

        dict_extra_layers = { # Extra layers to return for model visualisation
                            'blocks.1.res_blocks.0.branch1_conv': 'b0',
                            'blocks.1.res_blocks.0.add': 'b0_add',
                            'blocks.2.res_blocks.0.branch1_conv': 'b2',
                            'blocks.2.res_blocks.0.branch2.conv_a': 'b2_a',
                            'blocks.2.res_blocks.0.branch2.conv_b': 'b2_b',
                            'blocks.2.res_blocks.0.branch2.conv_c': 'b2_c',
                            'blocks.2.res_blocks.0.add': 'b2_add',
                            'blocks.3.res_blocks.1.add': 'b3_add',
                            }

        # Encoder to extract video feature
        if config['encoder'] == 'x3d_m':
            model = x3d_m(pretrained=True) # Requires input shape [batch, channels, frames, height, width] w/ frames divisible by 10
            dict_return_nodes = {'blocks.4.res_blocks.6.add':'raw_features'}
            if config['model_visualiser_mode']:
                dict_return_nodes.update(dict_extra_layers)
            self.encoder  = create_feature_extractor(model, return_nodes=dict_return_nodes)
            encoder_out_feature_size = 9408 # From flattening activation maps of shape [192, 7, 7]
        elif config['encoder'] == 'x3d_l':
            model = x3d_l(pretrained=True) # Requires input shape [batch, channels, frames, height, width] w/ frames divisible by 10
            dict_return_nodes = {'blocks.4.res_blocks.14.add':'raw_features'}
            if config['model_visualiser_mode']:
                dict_return_nodes.update(dict_extra_layers)
            self.encoder  = create_feature_extractor(model, return_nodes=dict_return_nodes)
            encoder_out_feature_size = 9408 # From flattening activation maps of shape [192, 7, 7]
        elif config['encoder'] == 'pre_extracted':
            logging.info('Pre_extracted encoder dataset used')
        else:
            raise ValueError('Invalid encoder type {pre_extracted}'.format(config['encoder']))

        if self.config['visual_mlp']:
            self.visual_mlp = VideoFeatureMLP(input_features=encoder_out_feature_size,
                                              output_features=config['visual_mlp_output_dim'],
                                              hidden_layers=config['visual_mlp_hidden_layers'],
                                              dropout=config['visual_mlp_dropout'])

        # Decoder GPT to generate corresponding text sequence
        decoder_input_size = encoder_out_feature_size if not self.config['visual_mlp'] else config['visual_mlp_output_dim']
        self.embedding = nn.Embedding(config['tokenizer_vocab_size'], decoder_input_size) # Translates token into fixed-size vectors that capture relations between words and semantics in a continuous space.
        self.positional_embedding = self.create_positional_embedding(config['max_tok_per_caption'], decoder_input_size)
        self.transformer_decoder = nn.TransformerDecoder(
                                                        nn.TransformerDecoderLayer(d_model=decoder_input_size,
                                                                                    nhead=config['decoder_heads'],
                                                                                    dim_feedforward=config['decoder_hidden_size'],
                                                                                    dropout=config['decoder_dropout']),
                                                                                    num_layers=config['decoder_layers']
                                                        )
        
        self.fc_out = nn.Linear(decoder_input_size, config['tokenizer_vocab_size'])
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def create_positional_embedding(max_seq_length, feature_size):
        """
        Create positional encoding for the input sequences.

        Args:
            max_seq_length (int): The maximum sequence length.
            feature_size (int): The size of the embeddings/features.

        Returns:
            Tensor: The positional encoding matrix.
        """
        pos_emb = torch.zeros(max_seq_length, feature_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_size, 2).float() * (-math.log(10000.0) / feature_size))
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        pos_emb = pos_emb.unsqueeze(0).transpose(0, 1)
        return pos_emb
    
    def forward(self, x, mask, captions, teacher_forcing_ratio): # Requires input in form [B, C=3, T, H, W]
        """
        Forward pass for the Transformer Decoder.

        Args:
            x (Tensor): The sequence of encoder's output.
            captions (Tensor): The target sequences for teacher forcing during training in token form.
            t (int): The current of timestep in the generation sequence.
            teacher_forcing_ratio (float): The probability of using teacher forcing.

        Returns:
            Tensor: The decoder's output logits.

        Tensor Shapes:
        1. Raw video shape: [batch, timesteps, height, width, channels]
        2. After encoder: [batch, timesteps, hidden_size, feature_size, feature_size]
        3. After flatten before mlp:[batch, timesteps, -1]
        4. After MLP: [timesteps, batch, MLP output dimension], timesteps first for the transformer decoder convention
        
        5. Decoder
        5.1 Positional embedding (pos_emb): [batch, pred_tokens, feature_dim]
        5.2 Caption embedding (caption_emb): [batch, pred_tokens, feature_dim]
        5.3 Target embedding (tgt_emb): [pred_tokens, batch, feature_dim]
        5.4 Output:  [pred_tokens, batch, feature_dim]
        5.5 Logit = [batch, timestep, vocab_size]
        5.6 Final logits = [batch, max_token_len, vocab_size]

        6. Next token selected from vocab size using greedy search
        next_token = self.softmax(logit[:, -1]).argmax(-1).unsqueeze(1) # Use predicted token as next input to decoder
        """
        # Encoder
        if self.config['encoder'] in ['x3d_m', 'x3d_l']: # If using x3d model and not pre-extracted features
            x = x.permute(0, 4, 1, 2, 3) # [B, T, H, W, C] -> [B, C, T, H, W]
            x = self.encoder(x) # Extract features from video frames
            x_act = {key: tensor.clone().cpu().detach() for key, tensor in x.items() if key != 'raw_features'} # Save the activations for visualisation
            x = x['raw_features'] # Shape after x3d [batch, hidden_size, timesteps, feature_size, feature_size)
            x = x.permute(0, 2, 1, 3, 4) # Permute to [batch, timesteps, hidden_size, feature_size, feature_size)
            x = torch.flatten(x, start_dim=2) # Flatten spatial dimension of feature map to [batch, timesteps, -1)
        else:
            x_act = x.clone().cpu().detach() # Save the activations for visualisation

        x = x.transpose(0, 1) # Transpose x to [timesteps, batch, feature_dim]
        mask =  ~mask.bool()   # Invert the mask for transformer as 1 means masked and 0 means not masked

        if self.config['visual_mlp']:
            x = self.visual_mlp(x) # Project features into token space and standardise length. (timesteps, batch, MLP output dimension), timesteps first for the transformer decoder convention

        # Decoder
        timesteps, batch_size, feature_size = x.shape # As required by transformer decoder
        
        if captions != None: # If training
            # Teacher Forcing. Start start-of-sequence token
            decoder_input = torch.full((self.config['batch_size'], 1), fill_value=self.config['start_token_idx'], dtype=torch.long).to(device)
            logits = torch.zeros([self.config['batch_size'], self.config['max_tok_per_caption'], self.config['tokenizer_vocab_size']], device=device)
            stop_token_detected = torch.zeros(self.config['batch_size'], dtype=torch.bool).to(device)

            # Prepare a boolean pattern for teacher forcing
            stochastic_pattern = []
            for _ in range(self.config['batch_size']):
                row = [random.random() < teacher_forcing_ratio for _ in range(self.config['max_tok_per_caption'])]
                stochastic_pattern.append(row)
            stochastic_pattern = torch.tensor(stochastic_pattern, dtype=torch.bool)
            tf_pattern = stochastic_pattern

            final_tf_pattern = []
            for t in range(1, self.config['max_tok_per_caption']): # Iterate over the length of caption length - start token
                # Compute the logits using the current decoder input
                pos_emb = self.positional_embedding # [max_tokens_per_captions, 1, feature_dim]
                pos_emb = pos_emb[:t, :] # Only get positional encoding up to current timestep -> [pred_tokens, 1, feature_dim]
                pos_emb = pos_emb.expand(-1, batch_size, -1) # Expand batch size and keep all the same -1 -> shape [pred_tokens, batch, feature_dim]
                pos_emb = pos_emb.transpose(0, 1).to(device) # Transpose to [batch, pred_tokens, feature_dim]
                
                caption_emb = self.embedding(decoder_input) # [batch, pred_tokens, feature_dim]
                tgt_emb = caption_emb + pos_emb # [batch, pred_tokens, feature_dim]
                tgt_emb = tgt_emb.permute(1,0,2) # [pred_tokens, batch, feature_dim]
                output = self.transformer_decoder(tgt=tgt_emb, memory=x, memory_key_padding_mask=mask) # [pred_tokens, batch, feature_dim]
                logit = self.fc_out(output).permute(1, 0, 2) # Output shape [batch, timestep, vocab_size]

                logits[:, t - 1, :] = logit[:, -1].squeeze(1) # Update latest logit prediction for next token
                use_teacher_forcing = tf_pattern[:, t - 1] # Decide the next input for the decoder using the deterministic pattern
                use_teacher_forcing = (use_teacher_forcing.sum() >= (use_teacher_forcing.size(0) / 2)) # if >= half == True
                final_tf_pattern.append(use_teacher_forcing.item()) # Save for output

                if use_teacher_forcing:
                    next_token = captions[:, t].unsqueeze(1) # Use ground truth token as next input to decoder
                else:
                    next_token =  self.softmax(logit[:, -1]).argmax(-1).unsqueeze(1) # Use predicted token as next input to decoder

                if t > 1: # Skip for the first token generation step
                    stop_token_detected |= (next_token.squeeze(1) == self.end_token_idx)
                    next_token[stop_token_detected] = 0 # Apply mask to set next token to zero if stop token detected

                decoder_input = torch.cat((decoder_input, next_token), dim=1) # Concat next token to the generated sequence
        else: # Inference, generate the sequence of logits that represent the predictions for each word in sequence over the vocab_size
            test_batch_size = x.shape[1] # Get batch size [timesteps, batch, feature_dim]
            decoder_input = torch.full((test_batch_size, 1), fill_value=self.config['start_token_idx'], dtype=torch.long).to(device)          
            logits = torch.zeros([test_batch_size, self.config['max_tok_per_caption'], self.config['tokenizer_vocab_size']], device=device)
            stop_token_detected = torch.zeros(test_batch_size, dtype=torch.bool).to(device)

            for t in range(1, self.config['max_tok_per_caption']): # Iterate over the length of caption length - start token
                # Compute the logits using the current decoder input
                pos_emb = self.positional_embedding # [max_tokens_per_captions, 1, feature_dim]
                pos_emb = pos_emb[:t, :] # Only get positional encoding up to current timestep -> [pred_tokens, 1, feature_dim]
                pos_emb = pos_emb.expand(-1, test_batch_size, -1) # Expand batch size and keep all the same -1 -> shape [pred_tokens, batch, feature_dim]
                pos_emb = pos_emb.transpose(0, 1).to(device) # Transpose to [batch, pred_tokens, feature_dim]
                
                caption_emb = self.embedding(decoder_input) # [batch, pred_tokens, feature_dim]
                tgt_emb = caption_emb + pos_emb # [batch, pred_tokens, feature_dim]
                tgt_emb = tgt_emb.permute(1,0,2) # [pred_tokens, batch, feature_dim]
                output = self.transformer_decoder(tgt=tgt_emb, memory=x, memory_key_padding_mask=mask)
                logit = self.fc_out(output).permute(1, 0, 2) # Output shape [batch, timestep, vocab_size]

                logits[:, t - 1, :] = logit[:, -1].squeeze(1) # Update latest logit prediction for next token

                next_token = logit[:, -1]
                if t != 1:
                    next_token[:, self.config['start_token_idx']] = 0

                next_token = self.softmax(next_token).argmax(-1).unsqueeze(1)

                if t > 1: # Skip for the first token generation step
                    stop_token_detected |= (next_token.squeeze(1) == self.end_token_idx)
                    next_token[stop_token_detected] = 0 # Apply mask to set next token to zero if stop token detected

                decoder_input = torch.cat((decoder_input, next_token), dim=1)

        return logits, x_act
