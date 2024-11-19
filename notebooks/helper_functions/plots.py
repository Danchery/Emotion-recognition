
import torch
import matplotlib.pyplot as plt
import os
import random
import warnings
import seaborn as sns
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def plot_curves(results: dict):
    """
    Plot Loss and Accuracy curves
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        Epochs = range(len(results['loss_train']))
        plt.figure(figsize=(9, 9))
        
        plt.subplot(2, 1, 1)
        sns.lineplot(x=Epochs, y=results['loss_train'], marker='o', label='loss_train')
        sns.lineplot(x=Epochs, y=results['loss_valid'], marker='x', label='loss_valid')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        
        plt.subplot(2, 1, 2)
        sns.lineplot(x=Epochs, y=results['acc_train'], marker='o', label='acc_train')
        sns.lineplot(x=Epochs, y=results['acc_valid'], marker='x', label='acc_valid')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.tight_layout()
        

def plot_predictions(dataset, model, device, num_images=3, classes=None):
    """
    As an example, it outputs 3 images with predicted and ground truth labels 
    """
    model.eval()
    plt.figure(figsize=(15, 6))

    images = random.sample(range(len(dataset)), num_images)

    for i, idx in enumerate(images):
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        predicted_label = predicted.item()
        image = image.squeeze(0).cpu()
        image = image.mean(dim=0, keepdim=True)
        image = image.permute(1, 2, 0)
        
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image.squeeze().cpu(), cmap='gray')
        plt.title(f'True: {classes[label]} | Pred: {classes[predicted_label]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    

def plot_ConfMatr_and_ClassReport(model, dataset_test, dataloader_test, device):
    """
    This function display Confusion matrix and classification report
    """
    pred_labels = []
    true_labels = dataset_test.targets
    
    model.eval()
        
    with torch.inference_mode():
            
        for X, y in dataloader_test:
    
            X, y = X.to(device), y.to(device)
    
            logits_test = model(X)
            y_test = torch.argmax(torch.softmax(logits_test, dim=1), dim=1)
            pred_labels.extend(y_test.cpu().tolist())
    
        ConfusionMatrixDisplay.from_predictions(y_true=true_labels, y_pred=pred_labels, display_labels=dataset_test.classes, cmap='Greys')
        print(classification_report(y_true=true_labels, y_pred=pred_labels))
