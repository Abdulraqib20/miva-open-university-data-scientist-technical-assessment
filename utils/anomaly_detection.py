import pandas as pd

def find_anomalous_learners(df, min_avg_completion=40, max_avg_completion=50):
    """
    Identifies learners considered anomalous based on their viewing patterns.

    Anomalous learners are defined as those who:
    1. Have an average video completion rate between min_avg_completion and
       max_avg_completion (inclusive).
    2. Have watched fewer videos than the total number of unique videos available
       in the dataset.

    Args:
        df (pd.DataFrame): The main DataFrame containing student_id,
                           video_title, and completion_rate_percent.
        min_avg_completion (float): The minimum average completion rate threshold.
        max_avg_completion (float): The maximum average completion rate threshold.

    Returns:
        pd.DataFrame: A DataFrame containing the student_id, avg_completion_rate,
                      and unique_videos_watched for the identified anomalous learners.
                      Returns an empty DataFrame if input is unsuitable or no anomalies found.
    """
    required_cols = ['student_id', 'video_title', 'completion_rate_percent']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame must contain columns: {required_cols}")
        return pd.DataFrame()

    if df.empty:
        print("Input DataFrame is empty.")
        return pd.DataFrame()

    # Calculate total number of unique videos in the dataset
    total_unique_videos = df['video_title'].nunique()
    if total_unique_videos == 0:
        print("No video titles found in the dataset.")
        return pd.DataFrame()
    print(f"Total unique videos in dataset: {total_unique_videos}")

    # Calculate average completion rate and unique videos watched per student
    learner_stats = df.groupby('student_id').agg(
        avg_completion_rate=('completion_rate_percent', 'mean'),
        unique_videos_watched=('video_title', 'nunique')
    ).reset_index()

    # Average completion rate within the specified range
    learners_in_range = learner_stats[
        (learner_stats['avg_completion_rate'] >= min_avg_completion) &
        (learner_stats['avg_completion_rate'] <= max_avg_completion)
    ]

    # Watched fewer videos than the total available
    anomalous_learners = learners_in_range[
        learners_in_range['unique_videos_watched'] < total_unique_videos
    ]

    print(f"Found {len(anomalous_learners)} potential anomalous learners.")

    return anomalous_learners[['student_id', 'avg_completion_rate', 'unique_videos_watched']]
