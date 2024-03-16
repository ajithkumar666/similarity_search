from sentence_transformers import SentenceTransformer
import numpy as np

# Function to calculate Euclidean distance
def _euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def euclidean_distance():
    # Encode all texts to get their embeddings
    all_texts = [input_text] + similar_texts
    embeddings = model.encode(all_texts)

    # Input text embedding
    input_embedding = embeddings[0]


    # Input text embedding
    input_embedding = embeddings[0]

    # Compute Euclidean distance between input text and each of the similar texts
    distances = [(similar_texts[i], _euclidean_distance(input_embedding, emb)) for i, emb in enumerate(embeddings[1:])]

    # Sort the texts by their Euclidean distance, closest first
    sorted_distances = sorted(distances, key=lambda x: x[1])

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

euclidean_distance()
