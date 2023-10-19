import pandas as pd
import torch
import numpy as np
import os
import re
import ast
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision.transforms import ToTensor
from preprocess import dpc_feature, aac_feature
from utils import load_tokenizer


class CustomDataset(Dataset):
    def __init__(self, csv_file, tok, configs, pretrained_weights=False):
        counts = csv_file['label'].value_counts()
        weights = np.array([1, max(counts) / min(counts)]).astype(np.float16)
        self.weights = torch.from_numpy(weights)
        self.class_weights = {0: 1, 1: max(counts) / min(counts)}
        self.df = csv_file[['window', 'label']]
        self.features = configs['features']
        self.features_dict = self.load_features(csv_file, configs['features'])
        self.manual_tokenizer = not pretrained_weights
        self.tokenizer = tok
        self.transform = ToTensor()
        self.window_size = configs['window_size']

    @staticmethod
    def load_features(df, features):
        pssm_list = []
        physiochemical_list = []
        aac_list = []
        dpc_list = []
        for row in df.itertuples():
            if features['pssm']:
                pssm_list.append(np.array(ast.literal_eval(row[2])).astype(np.float32))
            if features['physiochemical']:
                physiochemical_list.append(np.array(ast.literal_eval(row[3])).astype(np.float32))
            if features['aac']:
                aac_list.append(np.array(aac_feature.aac_compute(row[1])).astype(np.float32))
            if features['dpc']:
                dpc_list.append(np.array(dpc_feature.dpc_compute(row[1])).astype(np.float32))
        print('loading features finished')
        return {
            'pssm': np.array(pssm_list),
            'physiochemical': np.array(physiochemical_list),
            'aac': np.array(aac_list),
            'dpc': np.array(dpc_list),
        }

    def __len__(self):
        return len(self.df)

    @staticmethod
    def text_preprocessing(text):
        text = ' '.join(re.findall('.{1,1}', text))
        text = re.sub(r"[UZOB]", "X", text)
        text = re.sub(r"[/^]", "[PAD]", text)
        return text

    def text_to_sequence(self, text):
        text = ' '.join(re.findall('.{1,1}', text))
        sequence = []
        for token in text.split(' '):
            try:
                sequence.append(self.tokenizer[token])
            except KeyError:
                sequence.append(self.tokenizer['<oov>'])
        return sequence

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 0]
        if self.features['pssm']:
            pssm = self.features_dict['pssm'][idx]
        else:
            pssm = np.array([])
        if self.features['physiochemical']:
            physiochemical = self.features_dict['physiochemical'][idx]
        else:
            physiochemical = np.array([])

        if self.features['aac']:
            aac = np.repeat(self.features_dict['aac'][idx][np.newaxis, :], self.window_size, axis=0)
        else:
            aac = np.array([])

        if self.features['dpc']:
            dpc = np.repeat(self.features_dict['dpc'][idx][np.newaxis, :], self.window_size, axis=0)
        else:
            dpc = np.array([])

        label = self.df.iloc[idx, 1]
        weight = self.class_weights[label]

        if not self.manual_tokenizer:
            processed_text = self.text_preprocessing(text)
            encoded_text = self.tokenizer(processed_text, return_tensors='pt',
                                          return_attention_mask=False, return_token_type_ids=False)
        else:
            encoded_text = self.text_to_sequence(text)
            encoded_text = torch.tensor(np.squeeze(encoded_text), dtype=torch.int32)

        label = np.array(label)
        label = torch.tensor(label)

        weight = np.array(weight).astype(np.float32)
        if not self.manual_tokenizer:
            return torch.squeeze(encoded_text['input_ids']), label, weight
        else:
            return encoded_text, {'pssm': pssm,
                                  'physiochemical': physiochemical,
                                  'aac': aac,
                                  'dpc': dpc}, label, weight


if __name__ == '__main__':
    test_configs = {'features': {'pssm': True, 'physiochemical': True, 'dpc': True, 'aac': True},
                    'window_size': 7}

    data_path = 'S:/Programming/Ubiquitylation/data/train data/processed/valid_hybrid/7.csv'
    test_df = pd.read_csv(data_path)

    # tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    tokenizer = load_tokenizer('.')
    training_data = CustomDataset(test_df, tokenizer, test_configs)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=False)
    for i, f, l, w in train_dataloader:
        print(i)
