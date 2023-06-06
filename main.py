import gradio as gr
from sentence_transformers import SentenceTransformer
from utils.similarity import get_similar_items
import numpy as np
import pandas as pd
import markdown
import random

# Create title, description, and article strings
title = "Clothing Similarity Search 👕"
description = "**Transformer-based search engine** to fetch Amazon URLs for similar clothing items given a text description.\n\n**Data Collection**:\nTo scrape quality clothing data containing proper description and URL for the product, Apify's Amazon Product Scraper was used. The scraped data for various clothing categories was downloaded into a CSV file.\n\n**Data Cleaning**:\nPandas was used to clean and preprocess the text data by removing special characters, lowercasing, and applying text normalization techniques.\n\n**Making Embeddings**:\nSentence-transformers library was used to generate embeddings for the cleaned data using the [all-MiniLM-L6-v2](https://example.com/model-card) model. The embeddings were saved into a .npy file for faster similarity search retrieval.\n\n**Cosine Similarity**:\nCosine similarity was used to find the similarity between the query and the product embeddings.\n"

model = SentenceTransformer('model')
embeddings = np.load('data/embeddings.npy')
clothing_data = pd.read_csv('data/clothing_data_preprocessed.csv')

def getURL(text, top_k):
    # Call your function to retrieve similar item URLs
    similar_urls = get_similar_items(text, embeddings, clothing_data, top_k)
    return similar_urls

input_text = gr.components.Textbox(lines=1, label="Input Descriptiont")
input_top_k = gr.components.Slider(label="Number of Recommendations", minimum=1, maximum=10, step=1, default=5)
output_html = gr.outputs.HTML(label="Similar Items")

def process_text(text, top_k):
    urls = getURL(text, top_k)
    random.shuffle(urls)  # Shuffle the URLs for variety
    html_links = "<br>".join([f'<a href="{url}" target="_blank">{url}</a>' for url in urls])
    return f'<div style="padding: 20px">{html_links}</div>'

iface = gr.Interface(
    fn=process_text,
    inputs=[input_text, input_top_k],
    outputs=output_html,
    title=title,
    description=description,
    examples=[
        ["casual men's t-shirt", 3],
        ["stylish summer dress", 5],
        ["elegant evening gown", 7],
    ],
    theme="default",
    layout="vertical",
    interpretation="default",
    allow_flagging="never",
)

iface.launch()
