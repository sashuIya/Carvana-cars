import shutil
import torch
from tqdm import tqdm
from common_utils import AverageMeter
from torch import nn
from torch.utils.data import DataLoader


def train_epoch_step(model: nn.Module, criterion, optimizer, train_loader: DataLoader):
    losses = AverageMeter()
    # Set train mode.
    model.train()

    for inputs, targets in tqdm(train_loader, 'In epoch'):
        targets = targets.cuda(async=True)
        inputs_var = torch.autograd.Variable(inputs.cuda())
        targets_var = torch.autograd.Variable(targets.cuda())

        # Compute output:
        output = model(inputs_var)
        loss = criterion(output, targets_var)

        # Update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.data[0], inputs.size(0))

    return losses.avg


def validate(model: nn.Module, criterion, valid_loader: DataLoader):
    losses = AverageMeter()
    # Set evaluation mode.
    model.eval()

    for inputs, targets in tqdm(valid_loader, 'Validating...':
        targets = targets.cuda(async=True)
        inputs_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
        targets_var = torch.autograd.Variable(targets.cuda(), volatile=True)

        # Compute output:
        output = model(inputs_var)
        loss = criterion(output, targets_var)

        losses.update(loss.data[0], inputs.size(0))

    return losses.avg


def adjust_learning_rate(optimizer, train_losses):
    EPS = 1e-4
    prev_prev_loss, prev_loss, current_loss = train_losses[-3:]
    if abs(current_loss - prev_loss) < EPS and abs(prev_loss - prev_prev_loss) < eps:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            param_group['lr'] = lr * 0.3


def train(model: nn.Module,
          criterion,
          optimizer,
          num_epochs: int,
          train_loader: DataLoader,
          valid_loader: DataLoader):
    best_valid_loss = float('inf')
    train_losses = [1, 2, 3]

    for epoch in tqdm(range(num_epochs), 'Over epochs'):
        adjust_learning_rate(optimizer, train_losses)
        train_loss = train_epoch_step(model, criterion, optimizer, train_loader)
        train_losses.append(train_loss)
        valid_loss = validate(model, criterion, valid_loader)

        print(
            'Epoch: {}\nTrain loss: {}\nValidation loss: {}'.format(epoch, train_loss, valid_loss))

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
