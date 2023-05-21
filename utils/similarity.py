from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

clothing_data = pd.read_csv('data/clothing_data_preprocessed.csv')

model = SentenceTransformer('model')
embeddings = np.load('data/embeddings.npy')

def get_similar_items(query, embeddings, clothing_data, top_k=5):
    # Encode the query text
    query_embedding = model.encode([query], convert_to_tensor=True)
    # Compute similarity scores
    similarity_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    # Sort indices based on similarity scores
    sorted_indices = similarity_scores.argsort(descending=True)
    # Get the top-k most similar indices
    similar_indices = sorted_indices[:top_k].cpu().numpy()
    # Get the URLs of the top-k similar items
    similar_urls = clothing_data.loc[similar_indices, 'url'].tolist()

    return similar_urls

# Assuming you have the embeddings and clothing_data available
query = "Men's jeans black color"
similar_urls = get_similar_items(query, embeddings, clothing_data, top_k=5)
print(similar_urls)