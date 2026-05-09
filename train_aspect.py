import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from src.dataset import AspectDataset
from src.model_aspect import AspectModel
import torch.nn as nn
from tqdm import tqdm


df = pd.read_csv("./data/processed/aspect_dataset.csv")
train_df, val_df = train_test_split(df, test_size=0.2)

tk = AutoTokenizer.from_pretrained("vinai/phobert-base")

train_loader = DataLoader(AspectDataset(train_df, tk), batch_size=16, shuffle=True)
val_loader = DataLoader(AspectDataset(val_df, tk), batch_size=16)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AspectModel().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()

best_f1 = 0.0
for epoch in tqdm(range(5)):
    model.train()
    total_loss = 0
    for i, b in enumerate(train_loader):
        print(i)
        ids, mask, y = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["labels"].to(DEVICE)
        logits = model(ids, mask)
        loss = loss_fn(model(ids, mask), y)
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
            p = torch.sigmoid(model(ids, mask)).cpu().numpy()
            preds.extend((p>0.3).astype(int))
            labels.extend(b["labels"].numpy())

    f1 = f1_score(labels, preds, average="micro")
    print("F1:", f1)
    print(f"Train loss: {avg_loss:.4f}")
    print(f"Validation F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(
            {
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "f1": f1
            },
            "./checkpoint/best_aspect_model.pt"
        )
        print("Best model saved!")

print("\nTraining completed!")
print(f"Best F1: {best_f1:.4f}")