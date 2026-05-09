import torch
from torch.utils.data import Dataset

class AspectDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["text"].tolist()
        self.labels = df["labels"].apply(eval).tolist()
        self.tk = tokenizer

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        enc = self.tk(self.texts[index], truncation=True,
                      padding="max_length", max_length=128,
                      return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[index], dtype=torch.float)
        }
    

class ABSADataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["text"].tolist()
        self.aspects = df["aspect"].tolist()
        self.labels = df["sentiment"].tolist()
        self.tk = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index] + "</s> " + self.aspects[index]
        enc = self.tk(text, truncation=True,
                      padding="max_length", max_length=128,
                      return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[index], dtype=torch.float)
        }