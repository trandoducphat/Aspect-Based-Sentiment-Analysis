import torch
from transformers import AutoTokenizer

from src.model_aspect import AspectModel
from src.model_absa import ABSAModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "vinai/phobert-base"

ASPECT_LABELS = [
    "đồ ăn",
    "giá",
    "phục vụ",
    "không gian",
    "chất lượng",
    "vệ sinh",
    "tốc độ"
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

aspect_model = AspectModel().to(DEVICE)
aspect_checkpoint = torch.load("./checkpoint/best_aspect_model.pt",map_location=DEVICE)

aspect_model.load_state_dict(aspect_checkpoint["model_state_dict"])
aspect_model.eval()

absa_model = ABSAModel().to(DEVICE)
absa_checkpoint = torch.load("./checkpoint/best_absa_model.pt",map_location=DEVICE)
absa_model.load_state_dict(absa_checkpoint["model_state_dict"])
absa_model.eval()


def predict_aspects(text, threshold=0.3):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = aspect_model(ids, mask)
        probs = torch.sigmoid(logits)
    probs = probs.cpu().numpy()[0]
    aspects = []
    
    for i, p in enumerate(probs):
        if p > threshold:
            aspects.append(ASPECT_LABELS[i])
    
    return aspects


def predict_sentiment(text, aspect):
    combined = text + " </s> " + aspect
    enc = tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        pred = absa_model(ids, mask).item()
    
    return round(pred, 3)


def analyze_review(text):
    aspects = predict_aspects(text)
    results = []
    for asp in aspects:
        score = predict_sentiment(text, asp)
    
        results.append({
            "aspect": asp,
            "sentiment": score
        })

    return results


if __name__ == "__main__":
    while True:
        text = input("\nNhập review: ")
        result = analyze_review(text)
        print("\nKết quả:\n")

        for r in result:
            print(f"{r['aspect']} -> {r['sentiment']}")