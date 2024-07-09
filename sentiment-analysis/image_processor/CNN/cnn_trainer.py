import numpy as np
from transformers import AdamW
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch
from cnn_dataset import CNNDataset
from cnn_architecture import CNNArchitecture
import matplotlib.pyplot as plt
from image_processor.CNN.loss_module.loss_functions import NLLLoss

class CNNTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    
    def __init__(self):
        self.dataset = CNNDataset()
        self.dataset.prepare_dataset()

        self.model = CNNArchitecture(self.dataset.image_width, self.dataset.image_height)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.01)

        print(self.dataset.train_labels)
        print(len(self.dataset.train_labels))
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.dataset.train_labels), y=self.dataset.train_labels.numpy())
        print(f"Class Weights: {class_weights}")

        weights = torch.tensor(class_weights, dtype=torch.float)
        weights = weights.to(self.device)

        self.criterion = NLLLoss()

    def calculate_accuracy(self, preds, labels):
        pred_labels = np.argmax(preds, axis=1)
        correct = (pred_labels == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy

    def train(self):
        total_correct = 0
        total_examples = 0
        total_loss = 0
        total_preds = []

        self.model.train()

        for step, batch in enumerate(self.dataset.train_dataloader):
            if step % 50 == 0 and not step == 0:
                print(f"Batch {step} of {len(self.dataset.train_dataloader)}")

            batch = [r.to(self.device) for r in batch]
            inputs, labels = batch

            self.optimizer.zero_grad()

            preds = self.model(inputs)

            total_correct += (torch.argmax(preds, axis=1) == labels).sum().item()
            total_examples += labels.size(0)

            loss = self.loss_function(preds, labels)
            total_loss += loss.item()
            loss.backward()

            self.optimizer.step()

            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

        avg_loss = total_loss / len(self.dataset.train_dataloader)
        total_preds = np.concatenate(total_preds, axis=0)
        train_accuracy = total_correct / total_examples

        return avg_loss, total_preds, train_accuracy

    def evaluate(self):
        total_correct = 0
        total_examples = 0

        print("Evaluating...")

        self.model.eval()

        total_loss = 0
        total_preds = []

        for step, batch in enumerate(self.dataset.val_dataloader):
            batch = [t.to(self.device) for t in batch]
            inputs, labels = batch

            with torch.no_grad():   
                preds = self.model(inputs)

                total_correct += (torch.argmax(preds, axis=1) == labels).sum().item()
                total_examples += labels.size(0)

                loss = self.criterion(preds, labels)
                total_loss += loss.item()

                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)

        avg_loss = total_loss / len(self.dataset.val_dataloader)
        total_preds = np.concatenate(total_preds, axis=0)
        val_accuracy = total_correct / total_examples

        return avg_loss, total_preds, val_accuracy

    def test_model(self):
        with torch.no_grad():
            preds = self.model(self.dataset.test_images.to(self.device))
            preds = preds.detach().cpu().numpy()

        preds = np.argmax(preds, axis=1)
        print(classification_report(self.dataset.test_labels, preds, zero_division=1))

    def create_model(self):
        best_valid_loss = float('inf')

        train_losses = []
        valid_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(self.epochs):
            print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))
            
            train_loss, _, train_accuracy = self.train()
            train_accuracies.append(train_accuracy)
            
            valid_loss, _, val_accuracy = self.evaluate()
            val_accuracies.append(val_accuracy)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'CNN_model_10.pt')
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')

        self.test_model()

        plt.plot(range(1, self.epochs + 1), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, self.epochs + 1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()

        plt.plot(range(1, self.epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, self.epochs + 1), valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

trainer = CNNTrainer()
trainer.create_model()