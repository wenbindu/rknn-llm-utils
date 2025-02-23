import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Mapping
M = {0: '负向', 1: '中性', 2: '正向'}
# Load the tokenizer and model from pretrained model.
tokenizer = AutoTokenizer.from_pretrained('./cn_roberta_model_token')
model = AutoModelForSequenceClassification.from_pretrained(
    './cn_roberta_model', num_labels=3
)
model.eval()
# inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(text):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()


txt = 'It is a nice day.'
ret = predict(txt)
idx = np.argmax(ret)
