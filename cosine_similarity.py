from sentence_transformers import SentenceTransformer
import numpy as np

# Function to calculate cosine similarity
def _cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def cosine_similarity():
    # Encode all texts to get their embeddings
    all_texts = [input_text] + similar_texts
    embeddings = model.encode(all_texts)

    # Input text embedding
    input_embedding = embeddings[0]

    # Compute cosine similarity between input text and each of the similar texts
    similarities = [(similar_texts[i], _cosine_similarity(input_embedding, emb)) for i, emb in enumerate(embeddings[1:])]

    # Sort the texts by their cosine similarity, most similar first
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    print(sorted_similarities)
    # Print the texts sorted by cosine similarity
    print("Texts sorted by Cosine similarity to the input text:", input_text)
    print("")
    for r, (text, similarity) in enumerate(sorted_similarities):
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


cosine_similarity()
