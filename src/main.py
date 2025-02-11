import torch
import torch.optim.sgd
from torch.utils.data import DataLoader
from model import Unet
from dataloader.dataloader import CustomImageDataset 
from dataloader.transforms import Transforms
from train import ModelTrainer
from evaluate import ModelEvaluator
import matplotlib.pyplot as plt
import numpy as np

def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS (Apple Silicon GPU) for training")
        return torch.device("mps")  
    else:
        print("Using CPU for training")
        return torch.device("cpu")  

def main():
    num_epochs = 5
    batch_size = 1
    learning_rate = 1e-4
    device = get_device()
    class_names = ["Background", "Car"]

    train_transform = Transforms.get_train_transforms()
    val_transform = Transforms.get_test_transforms()
    test_transform = Transforms.get_test_transforms()

    train_dataset = CustomImageDataset("data/train", transform=train_transform)
    val_dataset = CustomImageDataset("data/val", transform=test_transform)  
    test_dataset = CustomImageDataset("data/test", transform=test_transform)  


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = Unet(3, 2).to(device) 
    #criterion = torch.nn.BCEWithLogitsLoss() 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = ModelTrainer(model, train_loader, val_loader, criterion, optimizer, device, epochs=num_epochs)
    trainer.train()
    
    
    trainer.plot_training_metrics()

    torch.save(model, "unet_emre.pth")  
    print(f"Model saved as unet_emre.pth")

    evaluator = ModelEvaluator(model, val_loader, criterion, device, class_names=class_names)
    evaluator.evaluate()
    evaluator.plot_metrics()
    #evaluator.plot_confusion_matrix()
    #evaluator.visualize_predictions()

if __name__ == "__main__":
    main()
