import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output') # output directory for plots
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_abandoned_videos(df, abandon_rate=50):
    """
    Identifies videos most frequently abandoned before reaching halfway.

    Abandonment is defined as completion rate below the specified threshold.

    Args:
        df (pd.DataFrame): DataFrame with columns 'video_title' and 'completion_rate_percent'.
        abandon_rate (float): Maximum completion rate to count as abandonment (default 50%).

    Returns:
        pd.DataFrame: DataFrame with 'video_title' and 'abandonment_count', sorted descending.
    """
    required_cols = ['video_title', 'completion_rate_percent']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame must contain columns: {required_cols}")
        return pd.DataFrame()

    if df.empty:
        print("Input DataFrame is empty.")
        return pd.DataFrame()

    # Filter for views abandoned before or at halfway
    abandoned_views = df[df['completion_rate_percent'] <= abandon_rate]

    if abandoned_views.empty:
        print(f"No views found with completion at or below {abandon_rate}%.")
        return pd.DataFrame()

    # Count abandonments per video
    abandonment_counts = (
        abandoned_views['video_title']
        .value_counts()
        .reset_index()
    )
    abandonment_counts.columns = ['video_title', 'abandonment_count']
    abandonment_counts = abandonment_counts.sort_values('abandonment_count', ascending=False)

    print(f"Found {len(abandonment_counts)} videos with abandonment at or before {abandon_rate}% completion.")
    return abandonment_counts


def plot_abandoned_videos(abandonment_counts, top_n=10, abandon_rate=50):
    """
    Plots the top N videos with the highest abandonment counts.

    Args:
        abandonment_counts (pd.DataFrame): Output from find_abandoned_videos.
        top_n (int): Number of top videos to display.
        abandon_rate (float): Threshold used for abandonment (for plot title).

    Returns:
        plotly.graph_objects.Figure or None: The bar chart figure.
    """
    if abandonment_counts.empty:
        print("Error: No data to plot.")
        return None

    top_videos = abandonment_counts.head(top_n).sort_values('abandonment_count')

    fig = px.bar(
        top_videos,
        x='abandonment_count',
        y='video_title',
        orientation='h',
        title=f'Top {top_n} Videos Abandoned Before {abandon_rate}% Completion',
        labels={
            'abandonment_count': 'Number of Abandonments',
            'video_title': 'Video Title'
        },
        text='abandonment_count'
    )
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        title_x=0.5,
        margin=dict(l=100, r=20, t=50, b=70)
    )
    fig.update_traces(textposition='outside')

    try:
        plot_path = os.path.join(OUTPUT_DIR, 'abandoned_videos_plot.png')
        fig.write_image(plot_path)
        print(f"Abandoned videos plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving abandoned videos plot: {e}")

    return fig
