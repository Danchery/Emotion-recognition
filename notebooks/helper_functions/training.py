
import torch
import torch.nn as nn
from tqdm.auto import tqdm

def train_step(model, dataloader_train, loss_fn, optimizer, device):
    """
    Trains the model in one epoch
    """
    model.train()
    loss_train, acc_train = 0, 0
    for X, y in dataloader_train:
        
        X, y = X.to(device), y.to(device)
        
        logits_train = model(X)
        loss = loss_fn(logits_train, y)
        loss_train += loss.item()
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        y_pred_train = torch.argmax(torch.softmax(logits_train, dim=1), dim=1)
        acc_train += (y_pred_train == y).sum().item() / len(logits_train)

    loss_train = loss_train / len(dataloader_train)
    acc_train = acc_train / len(dataloader_train)

    return loss_train, acc_train

def valid_step(model, dataloader_valid, loss_fn, device):
    """
    Validates the model in one epoch
    """
    model.eval()
    
    loss_valid, acc_valid = 0, 0
    
    with torch.inference_mode():
        
        for X, y in dataloader_valid:

            X, y = X.to(device), y.to(device)

            logits_valid = model(X)
            loss = loss_fn(logits_valid, y)
            loss_valid += loss.item()

            y_valid_train = torch.argmax(torch.softmax(logits_valid, dim=1), dim=1)
            acc_valid += (y_valid_train == y).sum().item() / len(logits_valid)

    loss_valid = loss_valid / len(dataloader_valid)
    acc_valid = acc_valid / len(dataloader_valid)

    return loss_valid, acc_valid

def test(model, dataloader_test, loss_fn, device):
    """
    test the model in one epoch
    """
    model.eval()
    
    loss_test, acc_test = 0, 0
    
    with torch.inference_mode():
        
        for X, y in dataloader_test:

            X, y = X.to(device), y.to(device)

            logits_test = model(X)
            loss = loss_fn(logits_test, y)
            loss_test += loss.item()

            y_test = torch.argmax(torch.softmax(logits_test, dim=1), dim=1)
            acc_test += (y_test == y).sum().item() / len(logits_test)

    loss_test = loss_test / len(dataloader_test)
    acc_test = acc_test / len(dataloader_test)

    return loss_test, acc_test


def train(model, epochs, dataloader_train, dataloader_valid, loss_fn, optimizer, scheduler, device):
    """
    This function trains and validate model for a given number of epochs through train_step 
    and valid_step functions. As a result return dictionary of the form:
    results = {
        'loss_train' :[], 
        'acc_train' : [],
        'loss_valid' : [],
        'acc_valid' : []
    """
    
    results = {
        'loss_train' :[], 
        'acc_train' : [],
        'loss_valid' : [],
        'acc_valid' : []
    }
    
    for epoch in tqdm(range(epochs)):
        loss_train, acc_train = train_step(model=model, 
                                         dataloader_train=dataloader_train,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         device=device)
        loss_valid, acc_valid = valid_step(model=model,
                                           dataloader_valid=dataloader_valid,
                                           loss_fn=loss_fn,
                                           device=device)
        print(f'Epoch: {epoch+1}\nLoss_train:{loss_train} | Loss_valid:{loss_valid}\nAcc_train:{acc_train} | Acc_valid:{acc_valid}')

        scheduler.step()
        
        results['loss_train'].append(loss_train)
        results['acc_train'].append(acc_train)
        results['loss_valid'].append(loss_valid)
        results['acc_valid'].append(acc_valid)

    return results

def init_weights(layer):
    if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)) and layer.weight.requires_grad:
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)
