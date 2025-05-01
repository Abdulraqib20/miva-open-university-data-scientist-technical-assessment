import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output') # output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_clustering_data(df):
    """
    Prepares the data for learner clustering by calculating average completion
    rate and number of unique videos watched per student.

    Args:
        df (pd.DataFrame): The main DataFrame containing student_id,
                           video_title, and completion_rate_percent.

    Returns:
        pd.DataFrame: A DataFrame indexed by student_id with features for clustering.
    """
    required_cols = ['student_id', 'video_title', 'completion_rate_percent']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame must contain columns: {required_cols}")
        return pd.DataFrame()

    if df.empty:
        print("Input DataFrame is empty.")
        return pd.DataFrame()

    learner_features = df.groupby('student_id').agg(
        avg_completion_rate=('completion_rate_percent', 'mean'),
        unique_videos_watched=('video_title', 'nunique')
    )
    learner_features = learner_features.dropna()

    if learner_features.empty:
        print("No valid learner data found after aggregation.")
        return pd.DataFrame()

    return learner_features


def find_optimal_clusters(scaled_data, max_k=10):
    """
    Determines the optimal number of clusters using Silhouette Analysis.

    Args:
        scaled_data (np.ndarray): The scaled feature data for clustering.
        max_k (int): The maximum number of clusters to test.

    Returns:
        int: The optimal number of clusters based on silhouette score.
    """
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)
        silhouette_scores.append(score)
        print(f"k={k}, silhouette score={score:.4f}")

    # Identify k with maximum silhouette score
    if silhouette_scores:
        optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    else:
        print("Could not compute any silhouette scores. Defaulting to k=3.")
        optimal_k = 3

    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), silhouette_scores, marker='o')
    plt.title('Silhouette Analysis for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    sil_plot_path = os.path.join(OUTPUT_DIR, 'silhouette_scores.png')
    plt.savefig(sil_plot_path)
    print(f"Silhouette analysis plot saved to {sil_plot_path}")
    plt.close()

    print(f"\nOptimal number of clusters determined to be : k =  {optimal_k}")
    return optimal_k


def perform_clustering(learner_features, n_clusters=None, max_k=10):
    """
    Performs K-Means clustering on the learner features.

    Args:
        learner_features (pd.DataFrame): DataFrame with features like
                                         avg_completion_rate and unique_videos_watched.
        n_clusters (int, optional): The number of clusters to create.
                                    If None, attempts to find optimal k via silhouette analysis.
                                    Defaults to None.
        max_k (int): Max clusters to check when finding optimal k.

    Returns:
        pd.DataFrame: The original learner_features DataFrame with an added
                      'cluster' column.
    """
    if learner_features.empty or learner_features.shape[0] < 2:
        print("Not enough data points for clustering.")
        return learner_features

    # Select and scale features
    features_to_scale = learner_features[['avg_completion_rate', 'unique_videos_watched']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_to_scale)

    # Determine the number of clusters
    if n_clusters is None:
        max_k = min(max_k, learner_features.shape[0])
        if max_k < 2:
            print("Need at least 2 data points to determine clusters. Defaulting to 1 cluster.")
            n_clusters = 1
        else:
            n_clusters = find_optimal_clusters(scaled_features, max_k=max_k)

    n_clusters = max(1, min(n_clusters, learner_features.shape[0]))

    # Perform K-Means clustering
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        learner_features['cluster'] = cluster_labels
        print(f"Clustering performed with {n_clusters} clusters.")
    except Exception as e:
        print(f"Error during clustering: {e}")

    return learner_features


def visualize_clusters(clustered_data):
    """
    Visualizes the learner clusters using Plotly.

    Args:
        clustered_data (pd.DataFrame): DataFrame with learner features and cluster labels.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    if 'cluster' not in clustered_data.columns or clustered_data.empty:
        print("Error: Invalid data for visualization.")
        return None

    fig = px.scatter(
        clustered_data.reset_index(),
        x='avg_completion_rate',
        y='unique_videos_watched',
        color='cluster',
        title='Learner Segments Based on Engagement Metrics',
        labels={
            'avg_completion_rate': 'Average Video Completion Rate (%)',
            'unique_videos_watched': 'Number of Unique Videos Watched'
        },
        hover_data=['student_id']
    )
    fig.update_layout(title_x=0.5)
    fig.update_traces(marker=dict(size=8, opacity=0.8))

    try:
        plot_path = os.path.join(OUTPUT_DIR, 'learner_clusters.png')
        fig.write_image(plot_path)
        print(f"Cluster visualization saved to {plot_path}")
    except Exception as e:
        print(f"Error saving cluster plot: {e}")

    return fig

