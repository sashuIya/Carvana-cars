import shutil
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import AverageMeter


def train_epoch_step(model: nn.Module, criterion, optimizer, train_loader: DataLoader):
    # Set train mode.
    model.train()

    for inputs, targets in train_loader:
        targets = targets.cuda(async=True)
        inputs_var = torch.autograd.Variable(inputs)
        targets_var = torch.autograd.Variable(targets)

        # Compute output:
        output = model(inputs_var)
        loss = criterion(output, targets_var)

        # Update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(model: nn.Module, criterion, valid_loader: DataLoader):
    losses = AverageMeter()
    # Set evaluation mode.
    model.eval()

    for inputs, targets in valid_loader:
        targets = targets.cuda(async=True)
        inputs_var = torch.autograd.Variable(inputs, volatile=True)
        targets_var = torch.autograd.Variable(targets, volatile=True)

        # Compute output:
        output = model(inputs_var)
        loss = criterion(output, targets_var)

        losses.update(loss.data[0], inputs.size(0))

    return losses.avg


def train(model: nn.Module,
          criterion,
          optimizer,
          num_epochs: int,
          *,
          train_loader: DataLoader,
          valid_loader: DataLoader):
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        train_epoch_step(model, criterion, optimizer, train_loader)
        valid_loss = validate(model, criterion, valid_loader)

        is_best = valid_loss < best_valid_loss
        best_valid_loss = min(best_valid_loss, valid_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'valid_loss': valid_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def save_checkpoint(state, is_best, filename='output/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'output/model_best.pth.tar')
