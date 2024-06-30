import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
from bs4 import BeautifulSoup
import re

class BertDataset():
    TEXT = 'text'
    LABEL = 'label'

    RANDOM_STATE = 2018
    TRAIN_PERCENTAGE = 0.7
    TEST_PERCENTAGE = 0.5

    OPTIMAL_LENGTH_PERCENTILE = 70

    train_labels = None
    val_labels = None
    test_labels = None

    train_dataloader = None
    val_dataloader = None

    test_seq = None
    test_mask = None
    test_y = None

    def __init__(self, path, dataset_from=None, dataset_to=None, text='text', label='label'):
        self.TEXT = text
        self.LABEL = label

        self.df = pd.read_csv(path, header=None, sep=";", names=['text', 'label'], encoding='utf-8')

        if (
            dataset_from is not None and
            dataset_to is not None and
            dataset_from < dataset_to and
            dataset_from >= 0 and dataset_from <= len(self.df) and
            dataset_to >= 0 and dataset_to <= len(self.df)
        ):
            self.df = self.df[dataset_from:dataset_to]

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def split_data(self):
        train_text, temp_text, train_labels, temp_labels = train_test_split(self.df[self.TEXT], self.df[self.LABEL],
                                                                            random_state=self.RANDOM_STATE,
                                                                            test_size=0.4,
                                                                            stratify=self.df[self.LABEL])

        test_text, val_text, test_labels, val_labels = train_test_split(temp_text, temp_labels,
                                                                        random_state=self.RANDOM_STATE,
                                                                        test_size=0.5,
                                                                        stratify=temp_labels)

        return train_text, val_text, test_text, train_labels, val_labels, test_labels

    def tokenize(self, train_text, val_text, test_text):
        self.optimal_sentence_length = 25

        tokens_train = self.tokenizer.batch_encode_plus(train_text.tolist(),
                                                        truncation=True,
                                                        pad_to_max_length=True,
                                                        max_length=self.optimal_sentence_length)

        tokens_val = self.tokenizer.batch_encode_plus(val_text.tolist(),
                                                      truncation=True,
                                                      pad_to_max_length=True,
                                                      max_length=self.optimal_sentence_length)

        tokens_test = self.tokenizer.batch_encode_plus(test_text.tolist(),
                                                       truncation=True,
                                                       pad_to_max_length=True,
                                                       max_length=self.optimal_sentence_length)

        k = 0
        print('Training Comments -->>', train_text.tolist()[k])
        print('\nInput Ids -->>\n', tokens_train['input_ids'][k])
        print('\nDecoded Ids -->>\n', self.tokenizer.decode(tokens_train['input_ids'][k]))
        print('\nAttention Mask -->>\n', tokens_train['attention_mask'][k])
        print('\nLabels -->>', self.df['label'][k])

        return tokens_train, tokens_val, tokens_test

    def convert_lists_to_tensors(self, tokens_train, tokens_val, tokens_test):
        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_y = torch.tensor(self.train_labels.tolist())

        val_seq = torch.tensor(tokens_val['input_ids'])
        val_mask = torch.tensor(tokens_val['attention_mask'])
        val_y = torch.tensor(self.val_labels.tolist())

        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        test_y = torch.tensor(self.test_labels.tolist())

        return train_seq, train_mask, train_y, val_seq, val_mask, val_y, test_seq, test_mask, test_y

    def find_optimal_sentence_length(self):
        rows_length = []
        for text in self.df:
            rows_length.append(len(text))

        arr = np.array(rows_length)
        return int(np.percentile(arr, self.OPTIMAL_LENGTH_PERCENTILE))

    def clean_text(self, text):
        text = re.sub(r'<[^>]+>', '', text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        return text

    def prepare_dataset(self):
        self.df[self.TEXT] = self.df[self.TEXT].apply(self.clean_text).tolist()
        print(self.df.head())

        train_text, val_text, test_text, self.train_labels, self.val_labels, self.test_labels = self.split_data()
        self.optimal_sentence_length = self.find_optimal_sentence_length()

        tokens_train, tokens_val, tokens_test = self.tokenize(train_text, val_text, test_text)

        train_seq, train_mask, train_y, val_seq, val_mask, val_y, self.test_seq, self.test_mask, self.test_y = self.convert_lists_to_tensors(tokens_train, tokens_val, tokens_test)

        batch_size = 32
        train_data = TensorDataset(train_seq, train_mask, train_y)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        val_data = TensorDataset(val_seq, val_mask, val_y)
        val_sampler = SequentialSampler(val_data)
        self.val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)