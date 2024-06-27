import yaml
import torch

# Config files
with open('config/init_config.yaml', encoding='utf-8') as f:
    INIT_CONFIG = yaml.load(f, Loader=yaml.FullLoader)
    
device = torch.device(INIT_CONFIG['target_gpu'] if torch.cuda.is_available() else 'cpu')