import streamlit as st
import pickle
import pandas as pd

# Load data
movies = pickle.load(open('model/movies.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)

    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write("👉", movie)