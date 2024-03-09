import torch.nn as nn

class BertArchitecture(nn.Module):

    def __init__(self, bert, dropout=0.1, input_size=768, hidden_size=32, weight_decay=0.01):
        super(BertArchitecture, self).__init__()

        #set the base model
        self.bert = bert

        #set the dropout layer
        self.dropout = nn.Dropout(dropout)

        #hidden layer
        self.hidden_layer = nn.Linear(input_size, hidden_size)

        self.hidden_layer2 = nn.Linear(hidden_size, 32)

        #set activation function for the hidden layer
        self.relu = nn.ReLU()

        #output layer
        self.output_layer = nn.Linear(32, 2)

        #set activation function for output layer
        # self.softmax = nn.LogSoftmax(dim=1)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, sent_id, attention_mask):
        #pass the inputs to the model  
        _, cls_token_representation = self.bert(sent_id, attention_mask, return_dict=False)

        #pass through the hidden layer
        x = self.hidden_layer(cls_token_representation)

        #apply activation function for hidden layer output
        x = self.relu(x)

        #apply dropout to eliminate nodes
        x = self.dropout(x)

        x = self.hidden_layer2(x)
        x = self.relu(x)
        x = self.dropout(x)

        #pass through the output layer
        x = self.output_layer(x)

        #apply activation function for output layer output
        # x = self.sigmoid(x)
        # x = self.softmax(x)

        return x
