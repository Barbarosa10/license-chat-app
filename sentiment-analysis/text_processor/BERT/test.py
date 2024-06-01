from bertarchitecture import BertArchitecture
from bert_dataset import BertDataset
from transformers import AutoModel, BertTokenizerFast
import torch

# Loading the model
Model = BertArchitecture(AutoModel.from_pretrained('bert-base-uncased'))
Model.load_state_dict(torch.load('saved_weights_sentiment_chat12.pt', map_location=torch.device('cpu')))
Tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def Get_sentiment_for_text(Review):
    global Model, Tokenizer
    # Convert Review to a list if it's not already a list
    if not isinstance(Review, list):
        Review = [Review]
 
    inputs = Tokenizer.batch_encode_plus(Review,
                                        padding=True,
                                        truncation=True,
                                        add_special_tokens = True,
                                        max_length =25,
                                        return_tensors="pt")
    # print(inputs)
    with torch.no_grad():
        outputs = Model(inputs['input_ids'], inputs['attention_mask'])
    # print([f"anger: {outputs[0][0]}", f"fear: {outputs[0][1]}", f"joy: {outputs[0][2]}",
    #  f"love: {outputs[0][3]}", f"sadness: {outputs[0][4]}", f"surprise: {outputs[0][5]}"])
    print(outputs)
 
     # Get predicted class
    predicted_class = torch.argmax(outputs, dim=1).item()

     # Convert predicted class to human-readable label
    # label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Define your label mapping
    label_mapping = {0: "Negative", 1: "Positive"}  # Define your label mapping

    # label_mapping = {0: "anger", 1: "fear", 2: "joy", 3: "love", 4: "sadness", 5: "surprise"}

    predicted_label = label_mapping[predicted_class]

    return predicted_label



# print(model)0



# while True :
#     Review = input("Introdu un test pentru a prezice un sentiment: ")
#     # Review ='''My cat died the other day. I will bury her tomorrow!'''
#     # Get_sentiment(Review, tokenizer, model)
#     print(Get_sentiment_for_text(Review))
