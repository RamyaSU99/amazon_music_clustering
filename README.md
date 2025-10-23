# amazon_music_clustering
🎵 Song Clustering Using Audio Features

Automatically group songs based on their audio characteristics using unsupervised machine learning techniques.

📘 Project Overview

With millions of songs available on streaming platforms, manually categorizing them into genres or moods is impractical.
This project aims to automatically group similar songs based on features like tempo, energy, danceability, and more — without any predefined labels.

By leveraging clustering algorithms, we can uncover hidden patterns in the music data and create meaningful clusters that may represent genres, moods, or styles.

💡 Business Use Cases
Use Case	Description
🎧 Personalized Playlist Curation	Automatically group similar songs to enhance playlist generation.
🔍 Improved Song Discovery	Recommend songs with similar audio profiles to users.
🎤 Artist Analysis	Help artists identify other songs with similar sound characteristics.
💼 Market Segmentation	Streaming platforms can use clusters to understand listener behavior and optimize promotions.
🧠 Approach

The project follows a step-by-step machine learning workflow:

1. Data Exploration & Preprocessing

Goal: Understand the dataset, clean it, and prepare it for modeling.
Steps:

Load the dataset (single_genre_artists.csv) into a Pandas DataFrame.

Explore data types, column names, missing values, and duplicates.

Drop unnecessary columns (track_name, artist_name, track_id).

Visualize feature distributions to understand their variation.

Normalize data using StandardScaler or MinMaxScaler for fair distance-based clustering.

2. Feature Selection

Goal: Choose relevant features that capture musical characteristics.

Recommended Features:

danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, duration_ms

These describe rhythm, mood, instrumentation, and overall energy — essential for clustering songs by sound similarity.

3. Dimensionality Reduction (Optional but Recommended)

Goal: Visualize clusters effectively by reducing dimensions.

Options:

PCA – Reduces dimensions while preserving variance.

t-SNE – Captures complex relationships, ideal for visualization.

⚠️ Note: Use reduced dimensions only for visualization, not for clustering input.

4. Clustering Techniques

Goal: Apply clustering algorithms to group similar songs.

🔹 Option A: K-Means Clustering

Simple and widely used.

Determine optimal k using:

Elbow Method (SSE vs. k plot)

Silhouette Score

Run KMeans(n_clusters=k) and add cluster labels to the dataset.

🔹 Option B: DBSCAN

Detects arbitrarily shaped clusters and noise/outliers.

Tune eps and min_samples carefully.

🔹 Option C: Hierarchical Clustering

Visualize cluster merging using a dendrogram.

Does not require predefining the number of clusters.

5. Cluster Evaluation & Interpretation

Goal: Assess clustering quality and interpret results.

Evaluation Metrics:

Metric	Description
Silhouette Score	Measures how close samples are to their own cluster vs. others.
Davies-Bouldin Index	Lower values indicate better separation.
Inertia (K-Means)	Measures compactness of clusters.

Interpretation Example:

Cluster 0: High energy, high danceability → “Party Tracks”

Cluster 1: Low energy, high acousticness → “Chill Acoustic”

6. Visualization

Goal: Present results clearly and intuitively.

Ideas:

2D scatter plots (PCA/t-SNE) with color-coded clusters.

Bar charts of mean feature values per cluster.

Heatmaps comparing feature averages across clusters.

Feature distribution plots (e.g., tempo, energy).

7. Final Analysis & Export

Goal: Deliver usable, interpretable results.

Add cluster labels to the main dataset.

Summarize top songs per cluster.

Export the final dataset to CSV for downstream use.

Write a summary explaining each cluster’s defining characteristics.

📊 Expected Results

By the end of the project, you will:

Identify distinct song clusters based on audio similarity.

Visualize and interpret what each cluster represents musically.

Generate insights for recommendation systems, playlist generation, or market segmentation.
