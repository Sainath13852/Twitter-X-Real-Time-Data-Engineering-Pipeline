import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

# Download tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

# -----------------------------
# LOAD GPT-2 MODEL
# -----------------------------
print("Loading AI detector model...\n")

model_name = "gpt2"

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.eval()

print("Model loaded successfully!\n")

# -----------------------------
# ENTER TEXT DIRECTLY
# -----------------------------
print("Paste your text below:\n")

text = ""

while True:
    try:
        line = input()
        text += line + "\n"
    except EOFError:
        break

# -----------------------------
# SPLIT INTO SENTENCES
# -----------------------------
sentences = sent_tokenize(text)

# -----------------------------
# AI DETECTION FUNCTION
# -----------------------------
scores = []

def calculate_perplexity(sentence):

    encodings = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            **encodings,
            labels=encodings["input_ids"]
        )

    loss = outputs.loss
    perplexity = torch.exp(loss)

    return perplexity.item()

# -----------------------------
# ANALYZE TEXT
# -----------------------------
print("\nAnalyzing AI content...\n")

for sentence in sentences:

    try:
        ppl = calculate_perplexity(sentence)

        # Convert perplexity to AI probability
        ai_score = max(0, min(100, 100 - ppl))

        scores.append(ai_score)

    except:
        scores.append(0)

# -----------------------------
# OVERALL AI PERCENTAGE
# -----------------------------
overall_ai = np.mean(scores)

print("=" * 50)

print(f"TOTAL AI CONTENT DETECTED: {overall_ai:.2f}%")

print("=" * 50)

# -----------------------------
# HUMAN / AI LABEL
# -----------------------------
if overall_ai > 70:
    print("Result: Highly AI Generated\n")

elif overall_ai > 40:
    print("Result: Mixed Human + AI Content\n")

else:
    print("Result: Mostly Human Written\n")

# -----------------------------
# SHOW SENTENCE SCORES
# -----------------------------
print("Sentence Analysis:\n")

for i, sentence in enumerate(sentences):

    print(f"Sentence {i+1}")

    print(f"AI Probability: {scores[i]:.2f}%")

    print(sentence)

    print("-" * 50)

# -----------------------------
# GRAPH
# -----------------------------
plt.figure(figsize=(12,6))

plt.plot(scores, marker='o')

plt.title("AI Content Detection Graph")

plt.xlabel("Sentence Number")

plt.ylabel("AI Probability (%)")

plt.ylim(0, 100)

plt.grid(True)

plt.show()