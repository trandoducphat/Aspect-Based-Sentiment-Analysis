import torch.nn as nn
from transformers import AutoModel
import torch


class ABSAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained("vinai/phobert-base")
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask):
        out = self.enc(ids, attention_mask=mask)
        x = out.last_hidden_state.mean(dim=1)
        return torch.tanh(self.fc(x))