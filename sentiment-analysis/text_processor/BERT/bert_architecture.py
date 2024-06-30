import torch.nn as nn
from text_processor.BERT.regularization_module.regularization_functions import *

class BertArchitecture(nn.Module):
    def __init__(self, bert, dropout_rate=0.1, input_size=768, hidden_size=32):
        super(BertArchitecture, self).__init__()

        self.bert = bert        
        # self.dropout = nn.Dropout(dropout_rate)       
        self.dropout = Dropout(dropout_rate)   
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        # self.relu = nn.ReLU()
        self.relu = ReLU()
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, sent_id, attention_mask):
        _, cls_token_representation = self.bert(sent_id, attention_mask, return_dict=False)

        x = self.hidden_layer(cls_token_representation)
        x = self.relu.forward(x)
        x = self.dropout.forward(x)

        x = self.hidden_layer2(x)
        x = self.relu.forward(x)
        x = self.dropout.forward(x)

        x = self.output_layer(x)

        return x