import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from src.dataset import ABSADataset
from src.model_absa import ABSAModel
import torch.nn as nn
from tqdm import tqdm

df = pd.read_csv("data/processed/absa_dataset.csv")
train_df, val_df = train_test_split(df, test_size=0.2)

tk = AutoTokenizer.from_pretrained("vinai/phobert-base")

train_loader = DataLoader(ABSADataset(train_df, tk), batch_size=16, shuffle=True)
val_loader = DataLoader(ABSADataset(val_df, tk), batch_size=16)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ABSAModel().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

best_mae = 999999
for epoch in tqdm(range(5)):
    model.train()
    total_loss = 0
    for i, b in enumerate(train_loader):
        print(i)
        ids, mask, y = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["label"].to(DEVICE)
        preds = model(ids, mask).squeeze()
        loss = loss_fn(preds, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    avg_loss = total_loss/len(train_loader)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in val_loader:
            ids, mask = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)
            p = model(ids, mask).squeeze().cpu().numpy()
            preds.extend(p)
            labels.extend(b["label"].numpy())

    mae = mean_absolute_error(labels, preds)
    print(f"Validation MAE: {mae:.4f}")
    print(f"Train loss: {avg_loss:.4f}")

    if mae < best_mae:
        best_mae = mae
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "mae": mae
            },
            "./checkpoint/best_absa_model.pt"
        )
        print("Best ABSA model saved!")

print("Training completed!")
print(f"Best MAE: {best_mae:.4f}")