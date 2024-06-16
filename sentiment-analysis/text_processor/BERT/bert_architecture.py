import torch.nn as nn

class BertArchitecture(nn.Module):
    """
    BertArchitecture class defines the architecture for the downstream task using a pre-trained BERT model.
    It adds a dropout layer, a hidden layer with ReLU activation, and an output layer for classification.
    """

    def __init__(self, bert, dropout=0.1, input_size=768, hidden_size=512):
        """
        Initializes the BertArchitecture.

        Args:
            bert (nn.Module): Pre-trained BERT model.
            dropout (float): Dropout rate.
            input_size (int): Size of the input features (default is 768 for BERT base).
            hidden_size (int): Size of the hidden layer.
        """
        super(BertArchitecture, self).__init__()

        self.bert = bert        
        self.dropout = nn.Dropout(dropout)        
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, sent_id, attention_mask):
        """
        Forward pass for the BertArchitecture.

        Args:
            sent_id (torch.Tensor): Tensor of input sentence IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.

        Returns:
            torch.Tensor: Output logits for classification.
        """
        _, cls_token_representation = self.bert(sent_id, attention_mask, return_dict=False)

        x = self.hidden_layer(cls_token_representation)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)

        return x
