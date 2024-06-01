import numpy as np
from transformers import AdamW
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import transformers
from transformers import AutoModel, BertTokenizerFast
import torch
import torch.nn as nn
from bert_dataset import BertDataset
from bert_architecture import BertArchitecture
import matplotlib.pyplot as plt
# from torchviz import make_dot

class BertTrainer:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    bert = AutoModel.from_pretrained('bert-base-uncased')

    # number of training epochs
    epochs = 10
    
    def __init__(self):
        # prepare dataset from fine-tuning
        self.dataset = BertDataset(path='./chat.txt', dataset_from=0, dataset_to=20000, text='text', label='label')
        self.dataset.prepare_dataset()

        # # freeze all parameters for fine-tuning
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # model for fine-tuning
        self.model = BertArchitecture(self.bert)
        # push to GPU if available
        self.model = self.model.to(self.device)

        #define optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 5e-5, weight_decay = 0.01) 
        #compute the class weights to address class imbalance
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.dataset.train_labels), y=self.dataset.train_labels)
        print(f"Class Weights: {class_weights}")

        # converting list of class weights to a tensor
        weights= torch.tensor(class_weights, dtype=torch.float)

        # push to GPU if available
        weights = weights.to(self.device)

        # define the loss function
        # self.cross_entropy  = nn.NLLLoss(weight=weights) 
        # self.cross_entropy  = nn.BCELoss(weight=weights, targets.float())
        self.cross_entropy  = nn.CrossEntropyLoss(weight=weights)

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
            # print result at every 50 bacthes
            if step % 50 == 0 and not step == 0:
                print(f"Batch {step} of {len(self.dataset.train_dataloader)}")

            batch = [r.to(self.device) for r in batch]

            sent_id, mask, labels = batch           

            self.model.zero_grad()

            preds = self.model(sent_id, mask)

            # graph = make_dot(preds, params=dict(self.model.named_parameters()))
            # graph.render("computation_graph")

            # Calculate accuracy
            total_correct += (torch.argmax(preds, axis=1) == labels).sum().item()
            total_examples += labels.size(0)

            loss = self.cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            # backward pass to calculate the gradients
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters
            self.optimizer.step()

            # model predictions are stored on GPU. So, push it to CPU
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
        # Existing code...
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
            if step % 50 == 0 and not step == 0:
                # elapsed = format_time(time.time() - t0)

                print(f"Batch {step} of {len(self.dataset.val_dataloader)}")


            batch = [t.to(self.device) for t in batch]

            sent_id, mask, labels = batch

            with torch.no_grad():   
                preds = self.model(sent_id, mask)

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
            preds = self.model(self.dataset.test_seq.to(self.device), self.dataset.test_mask.to(self.device))
            preds = preds.detach().cpu().numpy()


        # model's performance
        preds = np.argmax(preds, axis = 1)
        print(classification_report(self.dataset.test_y, preds, zero_division=1))

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
                torch.save(self.model.state_dict(), 'saved_weights_sentiment_chat1.pt')
            
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


trainer = BertTrainer()
trainer.create_model()
trainer.test_model()

