from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from utils.similarity import get_similar_items
import pandas as pd
import numpy as np

clothing_data = pd.read_csv('data/clothing_data_preprocessed.csv')
model = SentenceTransformer('model')
embeddings = np.load('data/embeddings.npy')

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/predict")
def getURL(query: Query):
    # Get the query from the request payload
    query_text = query.query
    # Call your function to retrieve similar item URLs
    similar_urls = get_similar_items(query_text, embeddings, clothing_data, 5)
    return {"similar_urls": similar_urls}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
