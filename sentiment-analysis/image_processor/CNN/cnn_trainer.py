import numpy as np
from transformers import AdamW
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from cnn_dataset import CNNDataset
from cnn_architecture import CNNArchitecture
import matplotlib.pyplot as plt
# from torchviz import make_dot

class CNNTrainer:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # number of training epochs
    epochs = 100
    learn_rate = 0.01
    
    def __init__(self):
        # prepare dataset from fine-tuning
        self.dataset = CNNDataset()
        self.dataset.prepare_dataset()


        self.model = CNNArchitecture(self.dataset.image_width, self.dataset.image_height)

        # push to GPU if available
        self.model = self.model.to(self.device)

        #define optimizer
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learn_rate)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 5e-5, weight_decay=0.01)

        print(self.dataset.train_labels)
        print(len(self.dataset.train_labels))
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.dataset.train_labels), y=self.dataset.train_labels.numpy())
        print(f"Class Weights: {class_weights}")

        weights= torch.tensor(class_weights, dtype=torch.float)

        # push to GPU if available
        weights = weights.to(self.device)

        # define the loss function
        self.cross_entropy  = nn.NLLLoss(weight=weights)

        #define criterion
        # self.cross_entropy  = nn.CrossEntropyLoss()

    def calculate_accuracy(self, preds, labels):
        pred_labels = np.argmax(preds, axis=1)
        correct = (pred_labels == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy

    def train(self):
        # Existing code...
        total_correct = 0
        total_examples = 0

        self.model.train()

        #init loss and accuracy
        total_loss, total_accuracy = 0, 0

        # empty list to save model predictions
        total_preds=[]

        for step, batch in enumerate(self.dataset.train_dataloader):
            if step % 50 == 0 and not step == 0:
                print(f"Batch {step} of {len(self.dataset.train_dataloader)}")

            batch = [r.to(self.device) for r in batch]
            inputs, labels = batch


            self.optimizer.zero_grad()

            preds = self.model(inputs)

            # Calculate accuracy
            total_correct += (torch.argmax(preds, axis=1) == labels).sum().item()
            total_examples += labels.size(0)

            loss = self.cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            loss.backward()

            self.optimizer.step()

            preds = preds.detach().cpu().numpy()

        total_preds.append(preds)

        # compute the training loss of the epoch
        avg_loss = total_loss / len(self.dataset.train_dataloader)
    
        # predictions are in the form of (no. of batches, size of batch, no. of classes).
        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds  = np.concatenate(total_preds, axis=0)

        train_accuracy = total_correct / total_examples

        #returns the loss and predictions
        return avg_loss, total_preds, train_accuracy


    def evaluate(self):
        total_correct = 0
        total_examples = 0

        print("Evaluating...")

        self.model.eval()

        #init loss and accuracy
        total_loss, total_accuracy = 0, 0

        # empty list to save model predictions
        total_preds=[]

        for step, batch in enumerate(self.dataset.val_dataloader):
            # print result at every 50 bacthes
            # if step % 50 == 0 and not step == 0:
            #     # elapsed = format_time(time.time() - t0)

            #     print(f"Batch {step} of {len(self.dataset.val_dataloader)}")


            batch = [t.to(self.device) for t in batch]

            inputs, labels = batch

            with torch.no_grad():   
                preds = self.model(inputs)

                # Calculate accuracy
                total_correct += (torch.argmax(preds, axis=1) == labels).sum().item()
                total_examples += labels.size(0)

                loss = self.cross_entropy(preds, labels)
                total_loss = total_loss + loss.item()

                preds = preds.detach().cpu().numpy()

                total_preds.append(preds)

        # compute the training loss of the epoch
        avg_loss = total_loss / len(self.dataset.val_dataloader)
    

        total_preds  = np.concatenate(total_preds, axis=0)

        val_accuracy = total_correct / total_examples

        #returns the loss and predictions
        return avg_loss, total_preds, val_accuracy

    def test_model(self):
        # get predictions for test data
        with torch.no_grad():
            preds = self.model(self.dataset.test_images.to(self.device))
            preds = preds.detach().cpu().numpy()


        # model's performance
        preds = np.argmax(preds, axis = 1)
        print(classification_report(self.dataset.test_labels, preds, zero_division=1))

    def create_model(self):
        # set initial loss to infinite
        best_valid_loss = float('inf')

        # empty lists to store training and validation loss of each epoch
        train_losses=[]
        valid_losses=[]

        train_accuracies = []
        val_accuracies = []

        #for each epoch
        for epoch in range(self.epochs):
            
            print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))
            
            #train model
            train_loss, _, train_accuracy= self.train()
            train_accuracies.append(train_accuracy)
            
            #evaluate model
            valid_loss, _, val_accuracy  = self.evaluate()
            val_accuracies.append(val_accuracy)
            
            #save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'CNN_model_8.pt')
            
            # append training and validation loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')

        self.test_model()

        # Plotting accuracy
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
# trainer.test_model()

