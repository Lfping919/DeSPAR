import numpy as np

def get_param_groups(model, stage=1):
    pretrained_params = []
    newly_added_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if stage == 1:
            if 'encoder' in name and 'dgr' not in name:
                pretrained_params.append(param)
            else:
                newly_added_params.append(param)
                
        elif stage == 2:
            if name.startswith(('encoder', 'norm1', 'norm2', 'norm3', 'norm4', 'dgr')):
                pretrained_params.append(param)
                
            elif name.startswith('decoder'):
                if 'gate' in name:
                    newly_added_params.append(param)
                else:
                    pretrained_params.append(param)
                    
            else:
                newly_added_params.append(param)
            
    return pretrained_params, newly_added_params

def cosine_scheduler(start_lr=1e-5, base_lr=1e-4, final_lr=5e-6, total_epochs=65, warmup_epochs=5):
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_lr, base_lr, warmup_epochs)
    else:
        warmup_schedule = np.array([], dtype=np.float32)

    anneal_epochs = max(0, total_epochs - warmup_epochs)
    if anneal_epochs > 0:
        progress = np.arange(anneal_epochs, dtype=np.float32) / float(anneal_epochs)
        anneal_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1.0 + np.cos(np.pi * progress))
    else:
        anneal_schedule = np.array([], dtype=np.float32)

    schedule = np.concatenate((warmup_schedule, anneal_schedule))
    return schedule