import os
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import chromadb
from chromadb.utils import embedding_functions

print("Fetching 20-newsgroups...")
dataset = fetch_20newsgroups(
    subset='all',
    remove=('headers', 'footers', 'quotes')
)

texts = dataset.data
labels = dataset.target
categories = dataset.target_names

# Minimal cleaning – remove empty/short docs
clean_texts = []
clean_labels = []
clean_indices = []
for i, text in enumerate(texts):
    text = text.strip()
    if len(text.split()) >= 20:  # remove near-empty posts
        clean_texts.append(text)
        clean_labels.append(labels[i])
        clean_indices.append(i)

print(f"Kept {len(clean_texts)} documents after cleaning")

# Embedding model – best small model on MTEB
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps" if __name__ == "__main__" else "cuda")
print("Encoding all documents...")
embeddings = embedder.encode(
    clean_texts,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True
)  # crucial for cosine

# Fit BERTopic with precomputed embeddings
print("Fitting BERTopic (fuzzy clustering)...")
topic_model = BERTopic(
    language="english",
    calculate_probabilities=True,
    verbose=True,
    nr_topics="auto",           # hierarchical reduction → coherent topics
    min_topic_size=80,          # tuned for nice granularity
)

topics, probs = topic_model.fit_transform(
    documents=clean_texts,
    embeddings=embeddings
)

print(f"Discovered {topic_model.get_topic_info().shape[0] - 1} topics (plus outliers)")

# Save topic model for inference
os.makedirs("topic_model", exist_ok=True)
topic_model.save("topic_model", save_embedding_model=embedder)

# Persist to Chroma
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="newsgroups",
    metadata={"hnsw:space": "cosine"}
)

# Add all documents if not already present
if collection.count() == 0:
    print("Adding documents to Chroma...")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=clean_texts,
        metadatas=[
            {
                "original_category": categories[label],
                "dominant_topic": int(topic),
                "original_index": clean_indices[i]
            }
            for i, (label, topic) in enumerate(zip(clean_labels, topics))
        ],
        ids=[f"doc_{i}" for i in range(len(clean_texts))]
    )
    print("Chroma DB built!")
else:
    print("Chroma DB already exists, skipping.")

print("Preparation complete! You can now run the API.")
