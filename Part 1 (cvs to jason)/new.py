import nltk
import matplotlib.pyplot as plt
import numpy as np
import os

from nltk.tokenize import sent_tokenize

# -----------------------------
# DOWNLOAD NLTK DATA
# -----------------------------
nltk.download('punkt')
nltk.download('punkt_tab')

# -----------------------------
# FILE PATH
# -----------------------------
file_path = r"D:\data science\data engineering\sample.txt"

# -----------------------------
# CHECK FILE
# -----------------------------
if not os.path.exists(file_path):

    print("ERROR: sample.txt not found")
    print("Please place sample.txt here:")
    print(file_path)

    exit()

# -----------------------------
# READ FILE
# -----------------------------
with open(file_path, "r", encoding="utf-8") as f:

    text = f.read()

print("\nFile loaded successfully!")

# -----------------------------
# SPLIT INTO SENTENCES
# -----------------------------
sentences = sent_tokenize(text)

if len(sentences) < 2:

    print("Please add more text inside sample.txt")

    exit()

# -----------------------------
# AI / PLAGIARISM SCORE
# -----------------------------
scores = []

for sentence in sentences:

    words = sentence.split()

    total_words = len(words)

    unique_words = len(set(words))

    # repetition-based score
    repetition_score = 1 - (unique_words / total_words)

    score = repetition_score * 100

    # keep score between 0-100
    score = max(0, min(100, score))

    scores.append(score)

# -----------------------------
# TOTAL SCORE
# -----------------------------
overall_score = np.mean(scores)

print("\n" + "="*50)

print(f"TOTAL AI / PLAGIARISM DETECTED: {overall_score:.2f}%")

print("="*50)

# -----------------------------
# RESULT LABEL
# -----------------------------
if overall_score > 70:

    print("Highly AI-like / repetitive content")

elif overall_score > 40:

    print("Moderate AI-like content")

else:

    print("Mostly human-written content")

# -----------------------------
# GRAPH
# -----------------------------
plt.figure(figsize=(10,5))

plt.plot(scores)

plt.title("AI / Plagiarism Detection Graph")

plt.xlabel("Sentence Number")

plt.ylabel("Similarity Percentage")

plt.ylim(0,100)

plt.grid(True)

plt.show()