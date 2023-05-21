import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


df = pd.read_csv('data/clothing_similarity_search.csv')
# Using DataFrame.apply() and lambda function
df["product"] = df['title'].fillna('') + df['description'].fillna('')

# Using DataFrame.copy() create new DaraFrame.
clothing_data = df[['url', 'product']].copy()

def preprocess_text(text):
	# Tokenize the text into individual words
	tokens = word_tokenize(text)
	tokens = [token.lower() for token in tokens]
	# Remove special characters and punctuation
	tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]
	# Remove stopwords
	stop_words = set(stopwords.words('english'))
	tokens = [token for token in tokens if token not in stop_words]
	# Lemmatize the tokens
	lemmatizer = WordNetLemmatizer()
	tokens = [lemmatizer.lemmatize(token) for token in tokens]
	# Join the tokens back into a single string
	preprocessed_text = ' '.join(tokens)
	return preprocessed_text

preprocessed_products = []
for index, row in clothing_data.iterrows():
	preprocessed_product = preprocess_text(row['product'])
	preprocessed_products.append(preprocessed_product)

# Add the preprocessed text to a new column in the clothing_data
clothing_data['preprocessed_product'] = preprocessed_products

clothing_data.to_csv('data/clothing_data_updated.csv')