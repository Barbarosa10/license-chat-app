import random
import typing
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer


import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

   
def text_cleaning(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    pattern = r"[^a-zA-Z0-9\s,']"
    text = re.sub(pattern, '', text)
    # print(text)
    return text

def split_data(df):
    train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['sentiment'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3, 
                                                                    stratify=df['sentiment'])

    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)
    return train_text, val_text, test_text, train_labels, val_labels, test_labels

def tokenize(train_text, val_text, test_text, df):
    #Tokenize and encode the data using the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, use_fast=False)

    max_len= 512
    # Tokenize and encode the sentences
    X_train_encoded = tokenizer.batch_encode_plus(train_text.tolist(),
                                                padding=True, 
                                                truncation=True,
                                                add_special_tokens = True,
                                                max_length = max_len,
                                                return_tensors='tf')

    X_val_encoded = tokenizer.batch_encode_plus(val_text.tolist(), 
                                                padding=True, 
                                                truncation=True,
                                                add_special_tokens = True,
                                                max_length = max_len,
                                                return_tensors='tf')

    X_test_encoded = tokenizer.batch_encode_plus(test_text.tolist(), 
                                                padding=True, 
                                                truncation=True,
                                                add_special_tokens = True,
                                                max_length = max_len,
                                                return_tensors='tf')

    k = 0
    print('Training Comments -->>',df['text'][k])
    print('\nInput Ids -->>\n',X_train_encoded['input_ids'][k])
    print('\nDecoded Ids -->>\n',tokenizer.decode(X_train_encoded['input_ids'][k]))
    print(tokenizer.convert_ids_to_tokens(X_train_encoded['input_ids'][k]))
    print('\nAttention Mask -->>\n',X_train_encoded['attention_mask'][k])
    print('\nLabels -->>',df['sentiment'][k])

    return X_train_encoded, X_val_encoded, X_test_encoded, tokenizer

   
def Get_sentiment(Review, Tokenizer, Model):
    # Convert Review to a list if it's not already a list
    if not isinstance(Review, list):
        Review = [Review]
 
    Input_ids, Token_type_ids, Attention_mask = Tokenizer.batch_encode_plus(Review,
                                                                             padding=True,
                                                                             truncation=True,
                                                                             add_special_tokens = True,
                                                                             max_length=512,
                                                                             return_tensors='tf').values()
    prediction = Model.predict([Input_ids, Token_type_ids, Attention_mask])
 
    # Use argmax along the appropriate axis to get the predicted labels
    pred_labels = tf.argmax(prediction.logits, axis=1)
 
    # Convert the TensorFlow tensor to a NumPy array and then to a list to get the predicted sentiment labels
    pred_labels = [label[i] for i in pred_labels.numpy().tolist()]
    return pred_labels


if __name__ == '__main__':
    # path = './imdb_dataset.csv'
    # df = pd.read_csv(path, names = ['text', 'sentiment'])
    # df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})
    # print(df.head())

    # df['cleaned_sentence'] = df['text'].apply(text_cleaning).tolist()

    # train_text, val_text, test_text, train_labels, val_labels, test_labels = split_data(df)

    # X_train_encoded, X_val_encoded, X_test_encoded = tokenize(train_text, val_text, test_text, df)

    # # Intialize the model
    # model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # # Compile the model with an appropriate optimizer, loss function, and metrics
    # optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    # model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # # Step 5: Train the model
    # history = model.fit(
    #     [X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']],
    #     train_labels,
    #     validation_data=(
    #     [X_val_encoded['input_ids'], X_val_encoded['token_type_ids'], X_val_encoded['attention_mask']],val_labels),
    #     batch_size=32,
    #     epochs=3
    # )

    # #Evaluate the model on the test data
    # test_loss, test_accuracy = model.evaluate(
    #     [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']],
    #     test_labels
    # )
    # print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

    # # Save tokenizer
    # tokenizer.save_pretrained('./Tokenizer')
    
    # # Save model
    # model.save_pretrained('./Model')

     
    # Load tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('./Tokenizer')
    
    # Load model
    bert_model = TFBertForSequenceClassification.from_pretrained('./Model')

    # pred = bert_model.predict(
	# [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']])

    # # pred is of type TFSequenceClassifierOutput
    # logits = pred.logits

    # # Use argmax along the appropriate axis to get the predicted labels
    # pred_labels = tf.argmax(logits, axis=1)

    # # Convert the predicted labels to a NumPy array
    # pred_labels = pred_labels.numpy()

    label = {
        1: 'positive',
        0: 'Negative',
        2: 'Neutral'
    }

    # # Map the predicted labels to their corresponding strings using the label dictionary
    # pred_labels = [label[i] for i in pred_labels]
    # Actual = [label[i] for i in test_labels]

    # print('Predicted Label :', pred_labels[:10])
    # print('Actual Label :', Actual[:10])

     
    # print("Classification Report: \n", classification_report(Actual, pred_labels))

    while True:
        Review = input("Introdu un test pentru a prezice un sentiment: ")
        print(Get_sentiment(Review, bert_tokenizer, bert_model))
    # Review ='''My cat died the other day. I will bury her tomorrow!'''
    # print(Get_sentiment(Review, bert_tokenizer, bert_model))

