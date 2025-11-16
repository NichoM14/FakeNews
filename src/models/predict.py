from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

MODEL_DIR = "model"  # where your trained model is stored

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # inference mode

label_map = {
    0: "REAL",
    1: "FAKE",
    2: "SATIRE"
}

def predict(text: str):
    # Encode input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    label = label_map.get(predicted_class_id, "UNKNOWN")
    
    return label

if __name__ == "__main__":
    while True:
        text = input("\nEnter text to classify (or 'quit'): ")
        if text.lower() == "quit":
            break
        
        label = predict(text)
        print(f"\nPrediction: {label}")
