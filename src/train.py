from sklearn.metrics import f1_score
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
from torch.utils.tensorboard import SummaryWriter

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, epochs=10, log_file="results/training_log.txt", save_dir="results"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader  
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.log_file = log_file
        self.save_dir = save_dir
        self.best_val_loss = float('inf') 
        self.writer = SummaryWriter() 

        if not os.path.exists(self.log_file) or os.stat(self.log_file).st_size == 0:
            with open(self.log_file, "w") as f:
                f.write("Epoch\tTrain_Loss\tTrain_Acc\tVal_Loss\tVal_Acc\n")
                f.flush()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def train(self):
        self.model.train()
        writer = SummaryWriter("results")
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", total=len(self.train_loader), ncols=100, leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs).float()
                if isinstance(outputs, dict):
                    outputs = outputs['out']  

                loss = self.criterion(outputs, labels)  
                loss.backward()
                self.optimizer.step()
                torch.mps.empty_cache()

                running_loss += loss.item()
                
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.numel()
                
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            epoch_loss = running_loss / len(self.train_loader)
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            epoch_acc = correct / total
            writer.add_scalar("Accuracy/train", epoch_acc, epoch)  
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            writer.flush()
            
            
            f1 = self.calculate_f1_score(all_labels, all_preds)
            writer.add_scalar("F1/train", f1, epoch)
            writer.flush()

            with open(self.log_file, "a") as f:
                f.write(f"{epoch+1}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t\n")

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, F1 Score: {f1:.4f}")

            val_loss, val_acc, val_f1 = self.validate()
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("F1/val", val_f1, epoch)
            writer.flush()

            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            with open(self.log_file, "a") as f:
                f.write(f"{epoch+1}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\t{val_f1:.4f}\n")
            
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation F1 Score: {val_f1:.4f}")

            if val_loss < self.best_val_loss:
                print(f"Validation loss improved! Saving model...")
                self.best_val_loss = val_loss
                model_save_path = os.path.join(self.save_dir, f"best_model_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), model_save_path)
            
            test_images = sorted(glob.glob("data/test/images/*.jpg"))
            test_masks = sorted(glob.glob("data/test/testmask/*.png")) 
            self.predict_and_visualize_multiple(test_images, test_masks)
            
        return self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()

                outputs = self.model(inputs).float()
                if isinstance(outputs, dict):
                    outputs = outputs['out']

                loss = self.criterion(outputs, labels)  
                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.numel()

                all_preds.append(predicted.cpu().numpy().astype(int))
                all_labels.append(labels.cpu().numpy().astype(int))


        val_loss /= len(self.val_loader)
        val_acc = correct / total
        val_f1 = self.calculate_f1_score(all_labels, all_preds)
        return val_loss, val_acc, val_f1

    def calculate_f1_score(self, all_labels, all_preds):
        all_labels = np.concatenate(all_labels).astype(int)
        all_preds = np.concatenate(all_preds).astype(int)

        all_labels = all_labels.flatten()
        all_preds = all_preds.flatten()

        print(f"Unique values in labels: {np.unique(all_labels)}")
        print(f"Unique values in predictions: {np.unique(all_preds)}")
        print(f"Labels shape: {all_labels.shape}, Predictions shape: {all_preds.shape}")

        assert np.all(np.isin(all_labels, [0, 1])), f"Unexpected label values: {np.unique(all_labels)}"
        assert np.all(np.isin(all_preds, [0, 1])), f"Unexpected prediction values: {np.unique(all_preds)}"

        return f1_score(all_labels, all_preds, average='macro')




    def predict_and_visualize_multiple(self, test_images, test_masks):
        self.model.eval()
    
        for image_path, mask_path in zip(test_images, test_masks):
            image = Image.open(image_path).convert("RGB")
            original_image = image.copy()
            image = self.transform(image).unsqueeze(0).to(self.device)
        
            ground_truth = Image.open(mask_path).convert("L")
            ground_truth = ground_truth.resize((128, 128), Image.NEAREST)
            ground_truth = np.array(ground_truth, dtype=np.uint8)
            ground_truth = (ground_truth > 0).astype(np.uint8)  

            with torch.no_grad():
                output = self.model(image)
        
            predicted_class = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
            predicted_background = (predicted_class == 0).astype(np.uint8)
            predicted_foreground = (predicted_class == 1).astype(np.uint8)
        
            self.visualize_masks(original_image, ground_truth, predicted_background, predicted_foreground)

            user_input = input("Press Enter to show next image, or type 'q' and Enter to quit visualization: ")
            if user_input.strip().lower() == 'q':
                print("Exiting visualization and resuming training...")
                break

        self.model.train()

    def visualize_masks(self, original_image, gt_mask, pred_background, pred_foreground):
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
    
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("GT Mask")
        axes[1].axis("off")
    
        axes[2].imshow(pred_background, cmap='gray')
        axes[2].set_title("Predicted Background")
        axes[2].axis("off")
    
        axes[3].imshow(pred_foreground, cmap='gray')
        axes[3].set_title("Predicted Foreground")
        axes[3].axis("off")
    
        combined = np.zeros((pred_background.shape[0], pred_background.shape[1], 3), dtype=np.uint8)
        combined[pred_background == 1] = [0, 0, 255]  
        combined[pred_foreground == 1] = [255, 0, 0] 

        axes[4].imshow(combined)
        axes[4].set_title("Combined Overlay")
        axes[4].axis("off")
    
        plt.show()

    def plot_training_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Training Loss", marker='o')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss", marker='x')
        plt.plot(range(1, len(self.train_accuracies) + 1), self.train_accuracies, label="Training Accuracy", marker='o')
        plt.plot(range(1, len(self.val_accuracies) + 1), self.val_accuracies, label="Validation Accuracy", marker='x')
        plt.title("Training and Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.show()
        