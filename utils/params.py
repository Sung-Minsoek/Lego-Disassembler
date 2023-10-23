import torch
import torch.nn as nn
import torch.optim as optim

params = {
    'optimizer'     : None,
    'loss_function' : None,
    'scheduler'     : None,
    'stats'         : [[0.5], [0.25]],
    'batch_size'    : 64,
    'worker'        : 4,
    'epochs'        : 50,
    'momentum'      : 0.9,
    'wd_decay'      : 0.0005,
    'lr'            : 0.0002,
    'train_size'    : 0.9,
    'valid_size'    : 0.1,
    'milestones'    : [],       # For Step lr scheduler
    'dropout'       : 0.5,      # For Dropout
    'patience'      : 10,        # For Early Stopping
    'device'        : 'cuda'    # 'cuda' = use cuda
}

def init_params(params):
    params['loss_function'] = nn.CrossEntropyLoss(label_smoothing=0.1)
    params['optimizer'] = optim.AdamW(params=model.parameters(), lr = params['lr'], weight_decay=params['wd_decay'])
    params['scheduler'] = optim.lr_scheduler.OneCycleLR(params['optimizer'], params['lr'], total_steps=params['epochs'] * len(trainloader))

    return params
