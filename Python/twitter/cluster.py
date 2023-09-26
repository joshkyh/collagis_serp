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
    dfr = pd.read_parquet('data/twitter/twitter_with_embeddings.parquet', engine='pyarrow')


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
    fig.write_image('visualization/elbow_method_twitter.png')

def get_cluster() -> pd.DataFrame:

    # Read the reviews DataFrame from a Parquet file
    dfr = pd.read_parquet('data/twitter/twitter_with_embeddings.parquet', engine='pyarrow')



    # Convert the 'embedding' column to a NumPy array
    embedding_array = np.stack(dfr['embedding'].to_numpy())


    # Run k-means clustering on original high-dimensional embeddings
    labels, centers, avg_cosine_distance = kmeans_cosine(embedding_array, n_clusters=25)

    # Add the cluster labels to the DataFrame
    dfr['cluster_label'] = labels

    return dfr


def max_diversity_sampling(x, N=5):
    # Compute the cosine similarity matrix
    similarity_matrix = 1 - cdist(x['embedding'].tolist(), x['embedding'].tolist(), metric='cosine')
    # Negate the similarity matrix because we are looking to maximize similarity
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
    # Select the top N diverse samples
    sampled_indices = row_ind[:N]
    return ' || '.join(x.iloc[sampled_indices]['text'])

if __name__ == '__main__':
    dfr = get_cluster()

    # Group by 'cluster_label' and apply max_diversity_sampling
    top_X_snippets_per_cluster = dfr.groupby('cluster_label').apply(max_diversity_sampling).reset_index()

    # Rename the columns for clarity
    top_X_snippets_per_cluster.columns = ['cluster_label', 'concatenated_snippets']

    print(top_X_snippets_per_cluster)

    top_X_snippets_per_cluster['theme'] = ''
    top_X_snippets_per_cluster['theme'].at[0] = 'Promotion of Betting Sources'
    top_X_snippets_per_cluster['theme'].at[1] = 'Short Reactions to Sports Betting'
    top_X_snippets_per_cluster['theme'].at[2] = 'Miscellaneous Comments and Media Links'
    top_X_snippets_per_cluster['theme'].at[3] = 'Detailed Horse Racing Insights'
    top_X_snippets_per_cluster['theme'].at[4] = 'Free Offers and Promotions'
    top_X_snippets_per_cluster['theme'].at[5] = 'Discussions on Sports Teams and Bets'
    top_X_snippets_per_cluster['theme'].at[6] = 'Critical or Negative Opinions on Betting'
    top_X_snippets_per_cluster['theme'].at[7] = 'Analysis and Queries on Horse Racing'
    top_X_snippets_per_cluster['theme'].at[8] = 'Betting on Specific Horses or Competitors'
    top_X_snippets_per_cluster['theme'].at[9] = 'Discussions on Sports Membership and Locations'
    top_X_snippets_per_cluster['theme'].at[10] = 'Greyhound Racing Tips and Stats'
    top_X_snippets_per_cluster['theme'].at[11] = 'Opinions on Sports and Betting Popularity'
    top_X_snippets_per_cluster['theme'].at[12] = 'Social Engagement with Betting Community'
    top_X_snippets_per_cluster['theme'].at[13] = 'Farewell and Good Luck Messages'
    top_X_snippets_per_cluster['theme'].at[14] = 'Criticism and Commentary on Betting Strategies'
    top_X_snippets_per_cluster['theme'].at[15] = ''
    top_X_snippets_per_cluster['theme'].at[16] = ''
    top_X_snippets_per_cluster['theme'].at[17] = 'Promotions for Fixed Betting Matches'
    top_X_snippets_per_cluster['theme'].at[18] = 'Banter and Sarcasm on Betting'
    top_X_snippets_per_cluster['theme'].at[19] = 'Individual Shoutouts and Personalities'
    top_X_snippets_per_cluster['theme'].at[20] = 'Racing Updates and Late Scratchings'
    top_X_snippets_per_cluster['theme'].at[21] = 'Predictions and Reactions on Winners'
    top_X_snippets_per_cluster['theme'].at[22] = 'Requests for Betting Markets and Odds'
    top_X_snippets_per_cluster['theme'].at[23] = 'Racing Team and Driver Changes'
    top_X_snippets_per_cluster['theme'].at[24] = 'Discussions Around the Everest Race'



    # Join the 'theme' column to the reviews DataFrame
    dfr = dfr.merge(top_X_snippets_per_cluster[['cluster_label', 'theme']], on='cluster_label')

    # Created_at Group by day of week and count the number of rows
    dfr['created_at'] = pd.to_datetime(dfr['created_at'])
    dfr['day_of_week'] = dfr['created_at'].dt.day_name()

    # Count the number of rows per day of week
    day_of_week_counts = dfr.groupby('day_of_week').size().reset_index()



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
                     hover_data=['text', 'theme'],
                     template='plotly_white')

    # Export
    fig.write_html('visualization/twitter_2d_tsne_clusters.html')



