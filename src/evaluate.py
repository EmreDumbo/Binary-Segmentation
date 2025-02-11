import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import torch.nn.functional as F

class ModelEvaluator:
    def __init__(self, model, test_loader, criterion, device, class_names):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.class_names = class_names
        self.losses = []

    def evaluate(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                if isinstance(outputs, dict):  
                    outputs = outputs['out']
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)  
                correct += (predicted == labels).sum().item()
                total += labels.numel()


                all_preds.extend(predicted.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        unique_preds = np.unique(all_preds)
        print("Unique predicted classes:", unique_preds)


        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

        test_loss = running_loss / len(self.test_loader)
        test_acc = correct / total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        for i, class_name in enumerate(self.class_names):
            if i < len(precision):  
                print(f"Precision '{class_name}': {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")

        self.losses.append(test_loss)
        self.plot_metrics(precision, recall, f1)

        return test_loss, test_acc, precision, recall, f1

    def plot_metrics(self, precision, recall, f1):
        num_classes = len(precision) 
        x = np.arange(num_classes)
        width = 0.25

        plt.figure(figsize=(10, 6))  
        plt.bar(x - width, precision, width, label='Precision', color='skyblue')
        plt.bar(x, recall, width, label='Recall', color='lightgreen')
        plt.bar(x + width, f1, width, label='F1 Score', color='salmon')

        plt.xticks(x, self.class_names[:num_classes], rotation=90)  
        plt.legend(), plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.title('Evaluation Metrics per Class')
        plt.tight_layout(), plt.show()
