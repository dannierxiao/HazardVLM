from prettytable import PrettyTable
from modules.training_loop.logging_funct import logging

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    logging.info(table)
    logging.info(f'Total Trainable Params: {total_params:e}')

    return total_params
