import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances_argmin_min
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def kmeans_cosine(X, n_clusters, max_iter=100, tol=1e-4):
    X_norm = normalize(X, axis=1, norm='l2')
    rng = np.random.RandomState(42)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    prev_avg_cosine_distance = np.inf
    for i in range(max_iter):
        labels, distances = pairwise_distances_argmin_min(X_norm, centers, metric='cosine')
        new_centers = np.array([X[labels == j].mean(0) for j in range(n_clusters)])
        avg_cosine_distance = np.mean(distances)
        if np.abs(prev_avg_cosine_distance - avg_cosine_distance) < tol:
            break
        prev_avg_cosine_distance = avg_cosine_distance
        centers = new_centers
    return labels, centers, avg_cosine_distance

def tune_cluster() -> pd.DataFrame:

    # Read the reviews DataFrame from a Parquet file
    dfr = pd.read_parquet('data/productreviews/dodo_embeddings.parquet', engine='pyarrow')


    # Convert the 'embedding' column to a NumPy array
    embedding_array = np.stack(dfr['embedding'].to_numpy())


    # Run k-means for different numbers of clusters and store average cosine distance
    n_clusters_range = range(3, 31)
    avg_cosine_distances = []

    n_clusters = 3
    for n_clusters in n_clusters_range:
        _, _, avg_cosine_distance = kmeans_cosine(embedding_array, n_clusters=n_clusters)
        avg_cosine_distances.append(avg_cosine_distance)



    # Create the plot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(n_clusters_range), y=avg_cosine_distances, mode='lines+markers'))

    fig.update_layout(
        title='Elbow Method for Optimal Number of Clusters',
        xaxis_title='Number of Clusters',
        yaxis_title='Average Cosine Distance',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )


    # Save the plot as a PNG file
    # Note: The 'write_image' function requires the 'kaleido' package, which we cannot install in this environment.
    # You can uncomment the following line to save the figure when running the code in your local environment.
    fig.write_image('visualization/elbow_method_dodo.png')

def get_cluster() -> pd.DataFrame:

    # Read the reviews DataFrame from a Parquet file
    dfr = pd.read_parquet('data/productreviews/dodo_embeddings.parquet', engine='pyarrow')



    # Convert the 'embedding' column to a NumPy array
    embedding_array = np.stack(dfr['embedding'].to_numpy())


    # Run k-means clustering on original high-dimensional embeddings
    labels, centers, avg_cosine_distance = kmeans_cosine(embedding_array, n_clusters=8)

    # Add the cluster labels to the DataFrame
    dfr['cluster_label'] = labels

    return dfr


def max_diversity_sampling(x, N=15):
    # Compute the cosine similarity matrix
    similarity_matrix = 1 - cdist(x['embedding'].tolist(), x['embedding'].tolist(), metric='cosine')
    # Negate the similarity matrix because we are looking to maximize similarity
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
    # Select the top N diverse samples
    sampled_indices = row_ind[:N]
    return ' || '.join(x.iloc[sampled_indices]['snippet'])

if __name__ == '__main__':
    dfr = get_cluster()

    # Group by 'cluster_label' and apply max_diversity_sampling
    top_X_snippets_per_cluster = dfr.groupby('cluster_label').apply(max_diversity_sampling).reset_index()

    # Rename the columns for clarity
    top_X_snippets_per_cluster.columns = ['cluster_label', 'concatenated_snippets']

    print(top_X_snippets_per_cluster)

    themes_str = '''Poor Customer Service and Connectivity Issues
    Mixed Experiences with Internet Speed
    Inconsistent Support and Lost Revenue
    Positive and Reliable Experience
    Generally Satisfied with Professional Help
    Essential and Life-Changing Service
    Occasional Dropouts but Competitive Pricing
    Exceptional Service Compared to Other Providers
    '''

    themes = themes_str.split('\n')
    # remove spaces in strings
    themes = [theme.strip() for theme in themes]

    # for loop to assign themes to each cluster
    top_X_snippets_per_cluster['theme'] = ''
    for i in range(len(themes)):
        top_X_snippets_per_cluster['theme'].at[i] = themes[i]






    # Join the 'theme' column to the reviews DataFrame
    dfr = dfr.merge(top_X_snippets_per_cluster[['cluster_label', 'theme']], on='cluster_label')


    # Plots
    embedding_array = np.stack(dfr['embedding'].to_numpy())

    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2)
    dfr['tsne'] = list(tsne.fit_transform(embedding_array))

    # Convert the t-SNE results to DataFrame columns for easier plotting
    tsne_results = np.array(dfr['tsne'].tolist())
    dfr['tsne_1'] = tsne_results[:, 0]
    dfr['tsne_2'] = tsne_results[:, 1]

    fig = px.scatter(dfr,
                     x='tsne_1',
                     y='tsne_2',
                     color='cluster_label',
                     title='Interactive Scatter Plot of t-SNE Clusters',
                     labels={'tsne_1': 't-SNE 1', 'tsne_2': 't-SNE 2'},
                     hover_data=['snippet', 'theme'],
                     template='plotly_white')

    # Export
    fig.write_html('visualization/dodo_2d_tsne_clusters.html')



