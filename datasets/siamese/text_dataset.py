import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import GPT2Tokenizer


class TextDataset(Dataset):
    def __init__(self, csv_file: str, delimiter: str = ";", encoding: str = "utf-8"):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file, delimiter=delimiter, encoding=encoding)
        data = self.df[["unique_id", "text1", "text2", "similarity"]]
        doc = []
        for _, i in data.iterrows():
            doc.append([i[['unique_id']].values[0], i[['text1']].values[0], i[['text2']].values[0],
                        1 if i[['similarity']].values[0] else 0])
        self.df = pd.DataFrame(doc, columns=["group_id", "text1", "text2", "similarity"])
        self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

    def __getitem__(self, index):
        row = self.df.iloc[[index]]
        return (
            self.tokenizer(row[['text1']].values[0], return_tensors='pt'),
            self.tokenizer(row[['text2']].values[0], return_tensors='pt'),
            torch.from_numpy(np.array([row[['similarity']].values[0]], dtype=np.float32))
        )

    def __len__(self):
        return len(self.df.values)
