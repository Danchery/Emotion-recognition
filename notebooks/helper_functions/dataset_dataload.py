
import torch
import torchvision

def create_datasets(train_dir, valid_dir, test_dir, transforms):
    """
    Creating datasets for training, testing and validation for given data
    """
    dataset_train = torchvision.datasets.ImageFolder(root=train_dir,
                                                     transform=transforms)
    dataset_valid = torchvision.datasets.ImageFolder(root=valid_dir,
                                                     transform=transforms)
    dataset_test = torchvision.datasets.ImageFolder(root=test_dir,
                                                     transform=transforms)
    return dataset_train, dataset_valid, dataset_test
    

def create_dataloaders(dataset_train, dataset_valid, dataset_test, batch_size=32, shuffle=True):
    """
    Creating dataloaders for training, testing and validation for given data
    """

    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=4,
                                                   pin_memory=True,
                                                   persistent_workers=True,
                                                   pin_memory_device='cuda')
    dataloader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                   batch_size=batch_size,
                                                   num_workers=4,
                                                   pin_memory=True,
                                                   persistent_workers=True,
                                                   pin_memory_device='cuda')
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                   batch_size=batch_size,
                                                   num_workers=4,
                                                   pin_memory=True,
                                                   pin_memory_device='cuda')
    
    return dataloader_train, dataloader_valid, dataloader_test
