import pandas as pd
import re

INPUT_PATH = "data/raw/foody_raw_reviews.csv"

OUT_ASPECT = "data/processed/aspect_dataset.csv"
OUT_ABSA = "data/processed/absa_dataset.csv"
TOREAD_ASP = "data/readable_processed/aspect_dataset.csv"
TOREAD_ABSA = "data/readable_processed/absa_dataset.csv"

ASPECTS = {
    "đồ ăn": ["bún","phở","cơm","bánh","nem","chả","đồ ăn"],
    "giá": ["giá","mắc","rẻ","đắt"],
    "phục vụ": ["phục vụ","nhân viên","chủ"],
    "không gian": ["quán","view","không gian"],
    "chất lượng": ["ngon","dở","tệ","ổn"],
    "vệ sinh": ["sạch","bẩn"],
    "tốc độ": ["chậm","nhanh","lâu"]
}

LABELS = list(ASPECTS.keys())

def clean_text(text):
    text = re.sub(r"^\d+(\.\d+)?\s*", "", str(text))
    text = text.lower()
    text = re.sub(r"[^\w\sáàảãạăâêôơưđ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_rating(r):
    try:
        r = float(r)
    except:
        r = 5
    return (max(0,min(10,r)) - 5)/5

def extract_aspects(text):
    res = []
    for asp, kws in ASPECTS.items():
        if any(kw in text for kw in kws):
            res.append(asp)
    return list(set(res))

def encode_labels(aspects):
    vec = [0]*len(LABELS)
    for a in aspects:
        if a in LABELS:
            vec[LABELS.index(a)] = 1
    return vec

def build():
    df = pd.read_csv(INPUT_PATH)

    texts, aspects, sentiments = [], [], []

    for _, row in df.iterrows():
        text = clean_text(row["text"])
        sentiment = normalize_rating(row["rating"])
        asp = extract_aspects(text)

        if len(asp) == 0:
            continue

        texts.append(text)
        aspects.append(asp)
        sentiments.append(sentiment)

    asp_df_out = pd.DataFrame({
        "text": texts,
        "aspects": aspects,
        "sentiment": sentiments
    })

    # aspect dataset
    asp_df_out["labels"] = asp_df_out["aspects"].apply(encode_labels)
    asp_df_out.to_csv(OUT_ASPECT, index=True)

    # absa dataset
    rows = []
    for _, r in asp_df_out.iterrows():
        for asp in r["aspects"]:
            rows.append({
                "text": r["text"],
                "aspect": asp,
                "sentiment": r["sentiment"]
            })

    absa_df_out = pd.DataFrame(rows)
    absa_df_out.to_csv(OUT_ABSA, index=True)


if __name__ == "__main__":
    build()