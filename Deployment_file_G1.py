

# IMPORTING LIBRARIES
import pickle
from pickle import load
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# TITLE AND CONTENT
st.title("SONGS RECOMMENDATION SYSTEM")
st.write("Similar songs based on popularity")
st.image("music.jpg", width=700)

# IMPORTING THE DATA
df = pd.read_csv("songs data set.csv")
df[df.duplicated()]
df_1=df.drop_duplicates()
df_new=df_1.drop(['song_duration_ms'], axis=1)

# SIDEBAR HEADER CONTENT
st.sidebar.image("kmeans.jpg", width=300)
st.sidebar.write("K-Means is one of the most widely used unsupervised clustering methods. Using clustering can address several known issues in recommendation systems, including increasing the diversity, consistency and reliability of recommendations.")

# SIDEBAR FOOTER CONTENT
st.sidebar.title("**ABOUT**")
st.sidebar.title("Made with streamlit")
st.sidebar.image("stlt.logo.png", width=180)
st.sidebar.header("P180 Group 1:")
st.sidebar.write("***Aji Thomas***",",","***AnanthaLakshmi Saripalle***")
st.sidebar.write("***Chimala Rajesh***",",","***Jagadeesh Korukonda***")
st.sidebar.write("***Margam Navya***",",","***Sanjusha Suresh***")
st.sidebar.header("Under the guidance of")
st.sidebar.write("***Neha Ramchandani***")
st.sidebar.write("***Himavanth Ila***")

# LOADING THE MODEL FROM DISK
loaded_model=pickle.load(open("songs.pkl",'rb'))

# CALCULATE THE COSINE SIMILARITY BETWEEN ALL THE SONGS
similarity_matrix = cosine_similarity(loaded_model)

def recommend_songs(song_name):
    # Find the index of the given song name in the feature matrix
    song_index = df_new[df_new["song_name"] == song_name].index[0]
    
    # Get the top 10 most similar songs to the given song name
    similar_songs = np.argsort(-similarity_matrix[song_index])[1:11]
    
    # Return the song names of the most similar songs
    return df_new.iloc[similar_songs]["song_name"]

w = st.text_input("Please type the song name for which you want the recommendation")
    
    
if st.button('Recommend songs'):
    st.write('Recommended songs are')
    forecast_model=recommend_songs(w)
    st.write(forecast_model)
    
    #st.success(forecast_model)