from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

print("Downloading SentenceTransformer model...")
SentenceTransformer("all-MiniLM-L6-v2")
print("Downloading tokenizer...")
AutoTokenizer.from_pretrained("bert-base-uncased")
print("Model downloads complete.")