"""Live headline classifier. Kept separate from the dashboard (app.py).
Loads a locally fine-tuned model from models/best_model/ (config + weights + tokenizer).
Nothing here runs until those files exist."""
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models" / "best_model"
LABELS = Path(__file__).parent / "data" / "label_mapping.json"


def model_ready():
    """Return (ok, message). Requires config, weights and tokenizer files."""
    if not MODEL_DIR.exists():
        return False, f"Folder not found: {MODEL_DIR}"
    names = {p.name for p in MODEL_DIR.iterdir()}
    weights = names & {"model.safetensors", "pytorch_model.bin"}
    tok = names & {"tokenizer.json", "tokenizer_config.json", "spiece.model", "vocab.txt", "sentencepiece.bpe.model"}
    missing = []
    if "config.json" not in names: missing.append("config.json")
    if not weights: missing.append("model.safetensors (or pytorch_model.bin)")
    if not tok: missing.append("tokenizer files")
    return (not missing), ("Ready" if not missing else "Missing: " + ", ".join(missing))


def load_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    id2label = {int(k): v for k, v in json.load(open(LABELS, encoding="utf-8"))["id2label"].items()}
    return tok, model, id2label


def predict(tok, model, id2label, text, max_length=128):
    import torch
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    return sorted(((id2label[i], p) for i, p in enumerate(probs)), key=lambda x: -x[1])
