import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv('data/movies.csv')

# Fill missing values
movies['genres'] = movies['genres'].fillna('')

# Convert text into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['genres']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Save model
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))
pickle.dump(movies, open('model/movies.pkl', 'wb'))

print("✅ Model trained and saved successfully!")