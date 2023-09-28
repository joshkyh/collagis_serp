import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances_argmin_min
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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
    reviews = pd.read_parquet('data/google_maps_reviews_with_embeddings.parquet', engine='pyarrow')


    # Read the google maps Datafram
    google_maps = pd.read_parquet('data/google_maps_results.parquet', engine='pyarrow')

    # Join google_maps into reviews
    reviews = reviews.merge(google_maps, on='data_id')

    # Convert the 'embedding' column to a NumPy array
    embedding_array = np.stack(reviews['embedding'].to_numpy())

    # Modify the kmeans_cosine function to return average cosine distance


    # Run k-means for different numbers of clusters and store average cosine distance
    n_clusters_range = range(2, 21)
    avg_cosine_distances = []

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
    fig.write_image('visualization/elbow_method.png')

def get_cluster() -> pd.DataFrame:

    # Read the reviews DataFrame from a Parquet file
    reviews = pd.read_parquet('data/google_maps_reviews_with_embeddings.parquet', engine='pyarrow')


    # Read the google maps Datafram
    google_maps = pd.read_parquet('data/google_maps_results.parquet', engine='pyarrow')

    # Join google_maps into reviews
    reviews = reviews.merge(google_maps, on='data_id')

    # Convert the 'embedding' column to a NumPy array
    embedding_array = np.stack(reviews['embedding'].to_numpy())


    # Run k-means clustering on original high-dimensional embeddings
    labels, centers, avg_cosine_distance = kmeans_cosine(embedding_array, n_clusters=8)

    # Add the cluster labels to the DataFrame
    reviews['cluster_label'] = labels

    return reviews

if __name__ == '__main__':
    reviews = get_cluster()

    # Group by 'cluster_label' and apply a lambda function to concatenate the top 10 'snippet' strings
    top_X_snippets_per_cluster = reviews.groupby('cluster_label')['snippet'].apply(
        lambda x: ' || '.join(x.head(20))
    ).reset_index()

    # Rename the columns for clarity
    top_X_snippets_per_cluster.columns = ['cluster_label', 'concatenated_snippets']

    print(top_X_snippets_per_cluster)

    top_X_snippets_per_cluster['theme'] = ''
    top_X_snippets_per_cluster['theme'].at[0] = 'Poor Service and Quality: Staff rudeness, undercooked food, inconsistency.'
    top_X_snippets_per_cluster['theme'].at[1] = 'Positive Staff Experience: Friendly, knowledgeable staff, excellent service.'
    top_X_snippets_per_cluster['theme'].at[2] = 'Betting Experience: Focus on TAB, different betting options, customer service.'
    top_X_snippets_per_cluster['theme'].at[3] = 'General Approval: Overall good, various compliments, minor issues.'
    top_X_snippets_per_cluster['theme'].at[4] = 'Comfortable Atmosphere: Quiet, roomy, big screens for sports, easy parking.'
    top_X_snippets_per_cluster['theme'].at[5] = 'Betting Mixed Reviews: Varying feedback on betting experience, staff quality.'
    top_X_snippets_per_cluster['theme'].at[6] = 'Pub and Food Delight: Delicious meals, great atmosphere, various options.'
    top_X_snippets_per_cluster['theme'].at[7] = 'Poor Management and Atmosphere: Closed, slow service, negativity, mismanagement.'

    # Join the 'theme' column to the reviews DataFrame
    reviews = reviews.merge(top_X_snippets_per_cluster[['cluster_label', 'theme']], on='cluster_label')
    reviews['snippet_short'] = reviews['snippet'].str[:50]

    #### Dendrogram
    import numpy as np
    import pandas as pd
    from sklearn.metrics import pairwise_distances
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.cluster.hierarchy import linkage, leaves_list
    import plotly.figure_factory as ff

    # Compute the pairwise distance matrix
    distance_matrix = pairwise_distances(reviews['embedding'].tolist(), metric='cosine')

    # Convert to similarity
    similarity_matrix = 1 - distance_matrix

    # Hierarchical clustering
    linked = linkage(similarity_matrix, method='complete')

    # Create the dendrogram
    fig = ff.create_dendrogram(linked, orientation='left')

    # Instead of extracting the dendrogram's leaf order, use the original index for snippets
    ordered_snippets = reviews['snippet_short'].tolist()

    # Update the layout to show the ordered snippets
    fig.update_layout(
        yaxis=dict(
            ticktext=ordered_snippets,
            showticklabels=True,
            tickangle=0,
            tickfont=dict(size=6)  # Adjust font size if necessary
        )
    )

    fig.update_layout(
        margin=dict(t=50, r=50, b=50, l=400),  # Adjust the 'l' value as required
    )

    fig.update_layout(width=5000, height=15000)  # Adjust as per your requirement
    #fig.write_image("visualization/dendrogram.pdf")
    fig.write_html("visualization/dendrogram.html")

    fig.show()


    #### Top themes by store
    # Group by stores to see stores that over-index (as a percentage and nominal) for each theme
    stores = reviews.groupby(['address', 'theme']).size().reset_index()
    stores.columns = ['address', 'theme', 'count']

    # Calculate the total count for each address
    stores['total_count'] = stores.groupby('address')['count'].transform('sum')

    # Calculate count_percentage
    stores['count_percentage'] = (stores['count'] / stores['total_count']) * 100

    # Find the index of the maximum count_percentage for each address
    idx = stores.groupby('address')['count_percentage'].idxmax()

    # Filter the DataFrame to only include these indices
    top_themes = stores.loc[idx].reset_index(drop=True)

    # Plot top theme for each address

    fig = px.scatter(top_themes,
                     x='total_count',
                     y='count_percentage',
                     color='theme',
                     title='Top Theme for each store',
                     labels={'total_count': 'Total Count', 'count_percentage': 'Count Percentage'},
                     hover_data=['address'],
                     template='plotly_white')

    # Export
    fig.write_html('visualization/top_themes.html')



    #### t-sne cloud
    embedding_array = np.stack(reviews['embedding'].to_numpy())

    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2)
    reviews['tsne'] = list(tsne.fit_transform(embedding_array))

    # Convert the t-SNE results to DataFrame columns for easier plotting
    tsne_results = np.array(reviews['tsne'].tolist())
    reviews['tsne_1'] = tsne_results[:, 0]
    reviews['tsne_2'] = tsne_results[:, 1]

    # rename rating_x to review_rating
    reviews = reviews.rename(columns={'rating_x': 'review_rating'})

    fig = px.scatter(reviews,
                     x='tsne_1',
                     y='tsne_2',
                     color='cluster_label',
                     title='Interactive Scatter Plot of t-SNE Clusters',
                     labels={'tsne_1': 't-SNE 1', 'tsne_2': 't-SNE 2'},
                     hover_data=['address', 'review_rating', 'snippet', 'theme', 'date'],
                     template='plotly_white')

    # Export
    fig.write_html('visualization/2d_tsne_clusters.html')



