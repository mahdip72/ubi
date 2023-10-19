import pandas as pd
import torch
import numpy as np
import os
import re
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision.transforms import ToTensor
from tokenizers.trainers import WordPieceTrainer


class CustomDataset(Dataset):
    def __init__(self, csv_file, tok, pretrained_weights=False):
        counts = csv_file['label'].value_counts()
        weights = np.array([1, max(counts) / min(counts)]).astype(np.float16)
        self.weights = torch.from_numpy(weights)
        self.class_weights = {0: 1, 1: max(counts) / min(counts)}
        self.df = csv_file[['window', 'label']]
        self.manual_tokenizer = not pretrained_weights
        self.tokenizer = tok
        self.transform = ToTensor()

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
            return encoded_text, label, weight


if __name__ == '__main__':
    data_path = '../data/processed/windowed_all_labels'
    train_path = os.path.join(data_path, 'train', '9.csv')
    train_df = pd.read_csv(train_path)

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    training_data = CustomDataset(train_df, tokenizer)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=False)
    for i, j in train_dataloader:
        print(i)
        print(j)
