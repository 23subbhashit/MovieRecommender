from django.shortcuts import render,redirect
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import math
import pickle
from django.views.decorators.csrf import csrf_protect
# Create your views here.
class CustomKMeans:
    def __init__(self, n_clusters, max_iter=10, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(self.random_state)
        # Initialize centroids using kmeans++
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X)
            # Update centroids based on the mean of data points in each cluster
            new_centroids = self._update_centroids(X, labels)
            # Check if centroids have converged
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
            self.labels = labels

    def predict(self, X):
        # Assign each data point to the nearest centroid
        return self._assign_labels(X)

    def inertia(self, X):
        # Calculate inertia (sum of squared distances of samples to their closest cluster center)
        labels = self.predict(X)
        distances = np.array([np.linalg.norm(X[i] - self.centroids[labels[i]]) ** 2 for i in range(len(X))])
        return np.sum(distances)

    def _initialize_centroids(self, X):
        centroids = [X[np.random.choice(X.shape[0])]]
        while len(centroids) < self.n_clusters:
            dist_sq = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            i = np.searchsorted(cumulative_probs, r)
            centroids.append(X[i])
        return np.array(centroids)

    def _assign_labels(self, X):
        # Assign each data point to the nearest centroid
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)

    def _update_centroids(self, X, labels):
        # Update centroids based on the mean of data points in each cluster
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else self.centroids[i] for i in range(self.n_clusters)])
        return new_centroids

def recommend_movies_to_new_user(ratings, cosine_sim, new_user_ratings, n=10):
    # Add the new user's ratings to the ratings matrix
    new_user_row = pd.Series(new_user_ratings, name='NewUser')
    ratings = ratings._append(new_user_row)

    # Compute the average rating given by the new user
    new_user_bias = sum(new_user_ratings.values()) / len(new_user_ratings)

    # Compute cosine similarity between the new user and existing users
    new_user_sim = custom_cosine_similarity(ratings.fillna(0))

    # Get the similarity values between the new user and existing users
    new_user_sim_values = new_user_sim[-1]  # Assuming the new user is the last row

    # Filter users with similarity less than 0.4
    filtered_similar_users = [(user_id, sim_value) for user_id, sim_value in enumerate(new_user_sim_values[:-1]) if sim_value >= 0.25]

    # Initialize a dictionary to store aggregated ratings from similar users
    aggregated_ratings = {}

    # Aggregate ratings from similar users
    for user_id, sim_value in tqdm(filtered_similar_users, desc="Aggregating ratings", total=len(filtered_similar_users)):
        similar_user_ratings = ratings.iloc[user_id]
        for movie, rating in similar_user_ratings.items():
            if pd.notna(rating) and movie not in new_user_ratings:
                if movie not in aggregated_ratings:
                    aggregated_ratings[movie] = []
                # Remove bias from the rating and add it to the aggregated ratings
                bias_corrected_rating = rating - ratings.mean(axis=1)[user_id]
                aggregated_ratings[movie].append(bias_corrected_rating * sim_value)

    # Calculate the average rating for each movie, divided by the sum of similarities
    avg_ratings = {}
    for movie, ratings in aggregated_ratings.items():
        average_rating = (sum(ratings) / sum(new_user_sim_values[:-1])) + new_user_bias
        # Round the rating to the nearest integer
        rounded_rating = math.ceil(average_rating) if average_rating - math.floor(average_rating) >= 0.5 else math.floor(average_rating)
        avg_ratings[movie] = rounded_rating

    # Sort movies by average rating in descending order
    sorted_avg_ratings = sorted(avg_ratings.items(), key=lambda x: x[1], reverse=True)

    # Return the top N recommended movies
    top_n_recommendations = sorted_avg_ratings[:n]
    return top_n_recommendations

def custom_cosine_similarity(matrix):
    """
    Compute cosine similarity manually for a matrix.

    Parameters:
    - matrix: 2D numpy array representing the data matrix

    Returns:
    - cosine_sim: 2D numpy array representing the cosine similarity matrix
    """
    dot_product = np.dot(matrix, matrix.T)
    norms = np.linalg.norm(matrix, axis=1)
    norms_matrix = np.outer(norms, norms)
    cosine_sim = dot_product / norms_matrix
    return cosine_sim
@csrf_protect
def main(request):
    new_user_ratings = {}
    if request.method == 'POST':
        for i in range(1, 6):
            movie_title = request.POST.get(f'movie_title{i}')
            rating = request.POST.get(f'rating{i}')
            # Process the data (e.g., save it to the database)
            # You can perform validation and save data as needed
            new_user_ratings[movie_title] = int(rating)
            print(f"Movie Title: {movie_title}, Rating: {rating}")
        moviedata = pd.read_csv("C:/Users/subbh/Downloads/movies (1).csv")
        userdata = pd.read_csv("C:/Users/subbh/Downloads/users (1).csv")
        ratingdata = pd.read_csv("C:/Users/subbh/Downloads/ratings (1).csv")
        merged_data = pd.merge(moviedata, ratingdata, left_on='ID', right_on='MovieID')
        ratings = merged_data.pivot_table(index='UserID', columns='Title', values='Rating')
        cosine_sim_custom = custom_cosine_similarity(ratings.fillna(0))

        # Get recommendations for the new user
        recommendations = recommend_movies_to_new_user(ratings, cosine_sim_custom, new_user_ratings)
        return render(request,'main/CF.html',{'recom': recommendations})
    else:
        return render(request, 'main/CFform.html')
def home(request):
    return render(request,'main/home.html')
@csrf_protect
def SvdK(request):
    new_user_ratings = {}
    if request.method == 'POST':
        for i in range(1, 6):
            movie_title = request.POST.get(f'movie_title{i}')
            rating = request.POST.get(f'rating{i}')
            # Process the data (e.g., save it to the database)
            # You can perform validation and save data as needed
            new_user_ratings[movie_title] = int(rating)
            print(f"Movie Title: {movie_title}, Rating: {rating}")
        # Redirect to a success page or render a different template
        movie_data = pd.read_csv('C:/Users/subbh/Downloads/movies (1).csv')
        user_data = pd.read_csv('C:/Users/subbh/Downloads/users (1).csv')
        rating_data = pd.read_csv('C:/Users/subbh/Downloads/ratings (1).csv')

        # Merge rating data with movie data
        print("Merging rating data with movie data...")
        merged_data = pd.merge(rating_data, movie_data, left_on='MovieID', right_on='ID')

        # Merge with user data using UserID
        print("Merging with user data using UserID...")
        merged_data = pd.merge(merged_data, user_data, on='UserID')

        # Pivot table of user ratings with movie titles as columns
        print("Creating user-item matrix...")
        user_ratings = merged_data.pivot_table(index='UserID', columns='Title', values='Rating').fillna(0)
        # Load kmeans object
        with open("C:/Users/subbh/Downloads/vt1n1.pkl", 'rb') as f:
            Vt_train = pickle.load(f)
        # Load clusters dictionary
        with open("C:/Users/subbh/Downloads/u1n1 (2).pkl", 'rb') as f:
            U_train = pickle.load(f)

        # Load kmeans object
        with open('C:/Users/subbh/Downloads/s1n1.pkl', 'rb') as f:
            S_train = pickle.load(f)

        # Load kmeans object
        with open("C:/Users/subbh/Downloads/vt1n1.pkl", 'rb') as f:
            Vt_train = pickle.load(f)

        print(U_train , S_train , Vt_train)
        # Apply KMeans clustering to cluster users based on reduced features
        kmeans = CustomKMeans(n_clusters=22)
        kmeans.fit(U_train)
        # Add cluster labels to user data
        cluster_labels_train = kmeans.predict(U_train)
        train_data_with_clusters = pd.DataFrame(data=U_train, columns=[f'Component_{i}' for i in range(U_train.shape[1])])
        train_data_with_clusters['Cluster'] = cluster_labels_train
        # New user ratings
        #new_user_ratings = {'Toy Story (1995)': 5, 'Jurassic Park (1993)': 4, 'Forrest Gump (1994)': 3}

        # Create a DataFrame for the new user
        new_user_data = pd.DataFrame(new_user_ratings, index=[0])

        # Convert movie titles to columns
        new_user_movies = pd.DataFrame(columns=user_ratings.columns)

        # Merge new user data with the movie DataFrame
        new_user_data_merged = pd.merge(new_user_movies, new_user_data, how='outer').fillna(0)

        # Reorder the columns of new_user_data_merged to match the order of user_ratings
        new_user_data_reordered = new_user_data_merged[user_ratings.columns]
        # Transform the new user data using SVD
        new_user_data_reduced = np.dot(new_user_data_reordered, Vt_train.T)

        # Find the cluster of the new user
        new_user_cluster = kmeans.predict(new_user_data_reduced)

        # Filter users in the same cluster
        users_same_cluster = train_data_with_clusters[train_data_with_clusters['Cluster'] == new_user_cluster[0]]

        # Get the indices of users in the same cluster
        user_indices_same_cluster = users_same_cluster.index

        # Filter ratings of users in the same cluster
        ratings_same_cluster = merged_data[merged_data['UserID'].isin(user_indices_same_cluster)]

        # Calculate the mean rating for each movie among users in the same cluster
        mean_ratings = ratings_same_cluster.groupby('Title')['Rating'].mean().sort_values(ascending=False)
        print(new_user_ratings)
        return render(request,'main/results.html', {'mean_ratings': mean_ratings.head(10)})
    else:
        return render(request, 'main/SVDKform.html')
    
