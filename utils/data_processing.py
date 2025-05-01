import pandas as pd

def add_churn_label(df, threshold=49):
    """
    Adds a binary churn label to the DataFrame based on video completion rate.

    Args:
        df (pd.DataFrame): DataFrame with a 'completion_rate_percent' column.
        threshold (float): The completion rate percentage below which a learner
                           is considered to have churned on a video.
                           Defaults to 49.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'churn' column
                      (1 if completion_rate_percent < threshold, 0 otherwise).
                      Returns the original DataFrame with a warning if the required
                      column is missing.
    """
    if 'completion_rate_percent' not in df.columns:
        print("Warning: 'completion_rate_percent' column not found. Cannot add churn label.")
        return df

    df['churn'] = df['completion_rate_percent'].apply(lambda x: 1 if x < threshold else 0)
    print(f"Churn label added (1 if completion < {threshold}%, 0 otherwise).")
    return df
