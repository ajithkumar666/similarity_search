from sentence_transformers import SentenceTransformer
import numpy as np

# Function to calculate Manhattan distance
def _manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))


def manhattan_distance():
    # Encode all texts to get their embeddings
    all_texts = [input_text] + similar_texts
    embeddings = model.encode(all_texts)

    # Input text embedding
    input_embedding = embeddings[0]

    # Compute Manhattan distance between input text and each of the similar texts
    distances = [(similar_texts[i], _manhattan_distance(input_embedding, emb)) for i, emb in enumerate(embeddings[1:])]

    # Sort the texts by their Manhattan distance
    sorted_distances = sorted(distances, key=lambda x: x[1])

    # Print the texts sorted by Manhattan distance
    # Print the texts sorted by Euclidean distance
    for r, (text, similarity) in enumerate(sorted_distances):
        print("Rank",r+1,": ",text,similarity)
        print("")

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Input text
input_text = "What are the three primary colors?"
# List of similar texts
similar_texts = [
     "What are the three primary colors?",
     "What are the three primary colors of light and why are they considered primary?",
     "Identify and list the three primary colors.",
     "What are the 3 primary colors?"
]

manhattan_distance()
