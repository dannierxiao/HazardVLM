import logging #  random for unique model name number
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
