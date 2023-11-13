# import pandas as pd
# import numpy as np
# import nltk
# from nltk.stem.snowball import SnowballStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import streamlit as st
# from PIL import Image

# # Load the dataset
# data = pd.read_csv('Datasets.csv')

# # Remove unnecessary columns
# # data = data.drop('id', axis=1)

# # Define tokenizer and stemmer
# stemmer = SnowballStemmer('english')
# def tokenize_and_stem(text):
#     tokens = nltk.word_tokenize(text.lower())
#     stems = [stemmer.stem(t) for t in tokens]
#     return stems

# # Create stemmed tokens column
# data['title'] = data['title'].fillna('')
# data['description'] = data['description'].fillna('')
# data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['title'] + ' ' + row['description']), axis=1)

# # Define TF-IDF vectorizer and cosine similarity function
# tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)
# def cosine_sim(text1, text2):
#     # tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
#     text1_concatenated = ' '.join(text1)
#     text2_concatenated = ' '.join(text2)
#     tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
#     return cosine_similarity(tfidf_matrix)[0][1]

# # Define search function
# def search_products(query):
#     query_stemmed = tokenize_and_stem(query)
#     data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
#     results = data.sort_values(by=['similarity'], ascending=False).head(10)[['productId','title', 'description']]
#     return results

# # web app
# img = Image.open('logo.png')
# st.image(img,width=600)
# st.title("Search Engine and Product Recommendation System ON Kitpot Data")
# query = st.text_input("Enter Product Name")
# sumbit = st.button('Search')
# if sumbit:
#     res = search_products(query)
#     st.write(res)
import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

# Load the dataset
data = pd.read_csv('Datasets.csv')

# Remove unnecessary columns
# data = data.drop('id', axis=1)

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')
def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Create stemmed tokens column
data['title'] = data['title'].fillna('')
data['description'] = data['description'].fillna('')
data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['title'] + ' ' + row['description']), axis=1)

# Define TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)

# Compute TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(data['stemmed_tokens'].apply(lambda x: ' '.join(x)))

# Define cosine similarity function using precomputed TF-IDF matrix
def cosine_sim(query, tfidf_matrix):
    query_tfidf = tfidf_vectorizer.transform([' '.join(tokenize_and_stem(query))])
    return cosine_similarity(query_tfidf, tfidf_matrix)[0]

# Define search function
def search_products(query, tfidf_matrix):
    similarities = cosine_sim(query, tfidf_matrix)
    data['similarity'] = similarities
    results = data.sort_values(by=['similarity'], ascending=False).head(10)[['productId','title', 'description']]
    product_ids_array = results['productId'].to_numpy()
    print(product_ids_array)
    return results

# web app
img = Image.open('logo.png')
st.image(img, width=600)
st.title("Search Engine and Product Recommendation System ON Kitpot Data")
query = st.text_input("Enter Product Name")
submit = st.button('Search')

if submit:
    res = search_products(query, tfidf_matrix)
    st.write(res)


# mongodb+srv://root1:1234@cluster0.bcqibzg.mongodb.net/product
# kitpotProdcutss