import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
sns.set_theme(style='white', palette='muted')
import os
from IPython.display import display

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data') # Data directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output') # Output Directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load data
def load_data(file_path):
    """Loads the video engagement data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

# Compute Average and Median Completion Rate per Video
def calculate_video_engagement(df):
    """Computes the average and median completion rate per video."""
    if 'video_title' not in df.columns or 'completion_rate_percent' not in df.columns:
        print("Error: Required columns ('video_title', 'completion_rate_percent') not found.")
        return None

    video_engagement = df.groupby('video_title')['completion_rate_percent'].agg(['mean', 'median'])
    video_engagement = video_engagement.rename(columns={'mean': 'avg_completion_rate', 'median': 'median_completion_rate'})
    video_engagement_sorted = video_engagement.sort_values('avg_completion_rate', ascending=False)
    print("\n Top 5 Average and Median Watch Duration Rates per Video (%):")
    display(video_engagement_sorted.head())
    return video_engagement_sorted

# Plot Top Videos
def plot_top_videos_by_total_duration(df, top_n=10):
    """
    Plots the top N most-watched videos based on total duration (approximated by
    sum of completion rates) using Plotly.

    Args:
        df: DataFrame with video engagement data
        top_n: Number of top videos to display

    Returns:
        Plotly figure object
    """
    if 'video_title' not in df.columns or 'completion_rate_percent' not in df.columns:
        print("Error: Required columns not found.")
        return None

    video_duration = df.groupby('video_title')['completion_rate_percent'].sum().reset_index()

    # Get top N videos by total duration
    top_videos = video_duration.sort_values('completion_rate_percent', ascending=False).head(top_n)
    top_videos = top_videos.sort_values('completion_rate_percent')

    # plot
    fig = px.bar(
        top_videos,
        x='completion_rate_percent',
        y='video_title',
        orientation='h',
        title=f'Top {top_n} Most-Watched Videos by Total Engagement Duration',
        labels={
            'completion_rate_percent': 'Total Completion Rate (sum of %)',
            'video_title': 'Video Title'
        },
        text_auto='.2s'
    )

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        title_x=0.5,
        margin=dict(l=100, r=20, t=50, b=70),
        xaxis_title="Total Completion Rate (sum of percentages)",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    )

    fig.update_traces(
        textposition='outside',
        textfont=dict(size=12),
        marker_color='rgb(41, 128, 185)'
    )

    fig.write_image(os.path.join(OUTPUT_DIR, 'top_10_videos_by_duration.png'))

    return fig
