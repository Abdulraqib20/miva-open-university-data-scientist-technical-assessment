import pandas as pd

def min_max_scale(series):
    """Applies Min-Max scaling to a pandas Series."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index) # Assigning a neutral score - 0.5
    return (series - min_val) / (max_val - min_val)

def calculate_learner_engagement_score(df, weight_completion=0.6, weight_unique_videos=0.4):
    """
    Computes a custom engagement score for each learner.

    The score is based on a weighted combination of the learner's average
    video completion rate and the number of unique videos they watched.

    Args:
        df (pd.DataFrame): DataFrame containing student_id, video_title,
                           and completion_rate_percent columns.
        weight_completion (float): The weight assigned to the normalized average completion rate.
        weight_unique_videos (float): The weight assigned to the normalized count of unique videos watched.

    Returns:
        pd.DataFrame: DataFrame with student_id, avg_completion_rate,
                      unique_videos_watched, and engagement_score, sorted by
                      engagement_score in descending order.
                      Returns an empty DataFrame if input is unsuitable.
    """
    required_cols = ['student_id', 'video_title', 'completion_rate_percent']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame must contain columns: {required_cols}")
        return pd.DataFrame(columns=['student_id', 'engagement_score', 'avg_completion_rate', 'unique_videos_watched'])

    if df.empty:
        print("Input DataFrame is empty.")
        return pd.DataFrame(columns=['student_id', 'engagement_score', 'avg_completion_rate', 'unique_videos_watched'])


    learner_stats = df.groupby('student_id').agg(
        avg_completion_rate=('completion_rate_percent', 'mean'),
        unique_videos_watched=('video_title', 'nunique')
    ).reset_index()

    # Normalize features btw 0 to 1 using manual Min-Max scaling
    learner_stats['norm_avg_completion'] = min_max_scale(learner_stats['avg_completion_rate'])
    learner_stats['norm_unique_videos'] = min_max_scale(learner_stats['unique_videos_watched'])

    # Calculate the weighted engagement score
    learner_stats['engagement_score'] = (
        learner_stats['norm_avg_completion'] * weight_completion +
        learner_stats['norm_unique_videos'] * weight_unique_videos
    )

    top_learners = learner_stats.sort_values('engagement_score', ascending=False)
    return top_learners[['student_id', 'engagement_score', 'avg_completion_rate', 'unique_videos_watched']]
