import torch

def get_scheduler(optimizer, train_loader=None, epochs=0):
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=epochs)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.5, verbose=True, min_lr=1e-5)
    return scheduler
