import numpy as np
from transformers import AdamW
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import transformers
from transformers import AutoModel, BertTokenizerFast
import torch
from bert_dataset import BertDataset
from bert_architecture import BertArchitecture
import matplotlib.pyplot as plt
from text_processor.BERT.loss_module.loss_functions import CrossEntropyLoss

class BertTrainer:
    """
    BertTrainer class handles training, evaluating, and testing a BERT model for text classification.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bert = AutoModel.from_pretrained('bert-base-uncased')
    
    epochs = 100
    
    def __init__(self):
        """
        Initializes the BertTrainer by setting up the dataset, model, optimizer, and loss function.
        """
        self.dataset = BertDataset(path='./chat.txt', dataset_from=0, dataset_to=20000, text='text', label='label')
        self.dataset.prepare_dataset()

        for param in self.bert.parameters():
            param.requires_grad = False

        self.model = BertArchitecture(self.bert)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.01)

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.dataset.train_labels), y=self.dataset.train_labels)
        print(f"Class Weights: {class_weights}")

        weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        # self.cross_entropy = nn.CrossEntropyLoss(weight=weights)

        self.criterion = CrossEntropyLoss()

    def calculate_accuracy(self, preds, labels):
        """
        Calculates accuracy of predictions.
        
        Args:
            preds (np.array): Predictions from the model.
            labels (torch.Tensor): Actual labels.
            
        Returns:
            accuracy (float): Accuracy of the predictions.
        """
        pred_labels = np.argmax(preds, axis=1)
        correct = (pred_labels == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy

    def train(self):
        """
        Trains the model for one epoch.
        
        Returns:
            avg_loss (float): Average loss over the training data.
            total_preds (np.array): Predictions on the training data.
            train_accuracy (float): Accuracy on the training data.
        """
        total_correct = 0
        total_examples = 0
        self.model.train()
        total_loss = 0
        total_preds = []

        for step, batch in enumerate(self.dataset.train_dataloader):
            if step % 50 == 0 and not step == 0:
                print(f"Batch {step} of {len(self.dataset.train_dataloader)}")

            batch = [r.to(self.device) for r in batch]
            sent_id, mask, labels = batch

            self.model.zero_grad()
            preds = self.model(sent_id, mask)

            total_correct += (torch.argmax(preds, axis=1) == labels).sum().item()
            total_examples += labels.size(0)

            loss = self.criterion(preds, labels)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

        avg_loss = total_loss / len(self.dataset.train_dataloader)
        total_preds = np.concatenate(total_preds, axis=0)
        train_accuracy = total_correct / total_examples

        return avg_loss, total_preds, train_accuracy

    def evaluate(self):
        """
        Evaluates the model on the validation dataset.
        
        Returns:
            avg_loss (float): Average loss over the validation data.
            total_preds (np.array): Predictions on the validation data.
            val_accuracy (float): Accuracy on the validation data.
        """
        total_correct = 0
        total_examples = 0

        print("Evaluating...")
        self.model.eval()
        total_loss = 0
        total_preds = []

        for step, batch in enumerate(self.dataset.val_dataloader):
            if step % 50 == 0 and not step == 0:
                print(f"Batch {step} of {len(self.dataset.val_dataloader)}")

            batch = [t.to(self.device) for t in batch]
            sent_id, mask, labels = batch

            with torch.no_grad():
                preds = self.model(sent_id, mask)
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
        """
        Tests the model on the test dataset and prints the classification report.
        """
        with torch.no_grad():
            preds = self.model(self.dataset.test_seq.to(self.device), self.dataset.test_mask.to(self.device))
            preds = preds.detach().cpu().numpy()

        preds = np.argmax(preds, axis=1)
        print(classification_report(self.dataset.test_y, preds, zero_division=1))

    def create_model(self):
        """
        Trains and evaluates the model over a number of epochs, saving the best model weights.
        """
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
                torch.save(self.model.state_dict(), 'saved_weights_sentiment_chat1.pt')
            
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

trainer = BertTrainer()
trainer.create_model()
trainer.test_model()
