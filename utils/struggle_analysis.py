import pandas as pd
import plotly.express as px
import os
from IPython.display import display

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output') # output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_video_struggle(df, struggle_std_threshold=1.5):
    """
    Calculates a struggle score for each student-video interaction based on
    deviation from the video's average completion rate.

    Args:
        df (pd.DataFrame): DataFrame with student_id, video_title, completion_rate_percent.
        struggle_std_threshold (float): Number of standard deviations below the video's
                                        mean completion rate to be flagged as significant struggle.

    Returns:
        pd.DataFrame: Original DataFrame with added columns:
                      - video_avg_completion
                      - video_std_completion
                      - struggle_deviation (video_avg - student_completion)
                      - significant_struggle (boolean flag)
                      Returns an empty DataFrame on error.
    """
    required_cols = ['student_id', 'video_title', 'completion_rate_percent']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame must contain columns: {required_cols}")
        return pd.DataFrame()

    if df.empty:
        print("Input DataFrame is empty.")
        return pd.DataFrame()

    print("Calculating video completion statistics...")
    video_stats = df.groupby('video_title')['completion_rate_percent'].agg(['mean', 'std']).reset_index()
    video_stats.columns = ['video_title', 'video_avg_completion', 'video_std_completion']

    # Fill std NaN with 0 (for videos watched only once)
    video_stats['video_std_completion'] = video_stats['video_std_completion'].fillna(0)
    df_merged = pd.merge(df, video_stats, on='video_title', how='left')

    # Calculate struggle deviation
    df_merged['struggle_deviation'] = df_merged['video_avg_completion'] - df_merged['completion_rate_percent']

    # Flag significant struggle
    df_merged['significant_struggle'] = (
        (df_merged['struggle_deviation'] > 0) &
        (df_merged['completion_rate_percent'] < (df_merged['video_avg_completion'] - struggle_std_threshold * df_merged['video_std_completion']))
    )

    print("Struggle metrics calculated.")
    print(f"Instances of significant struggle (deviation > {struggle_std_threshold} * video_std): {df_merged['significant_struggle'].sum()}")

    return df_merged


def analyze_struggle_patterns(struggle_df, min_course_views=100):
    """
    Analyzes patterns in the struggle data and visualizes results:
    - Top learners by struggle count
    - Top videos by struggle count
    - Struggle rate by course (filtered and visualized)
    - Distribution of struggle deviations

    Args:
        struggle_df (pd.DataFrame): DataFrame output from calculate_video_struggle.
        min_course_views (int): Minimum number of views per course to include in course-level analysis.

    Returns:
        dict: Results and figures.
    """
    if struggle_df.empty or 'significant_struggle' not in struggle_df.columns:
        print("Error: Invalid input DataFrame for struggle pattern analysis.")
        return {}

    results = {}
    print("\nAnalyzing Struggle Patterns...")

    # Learners with most significant struggle instances
    struggling_learners = struggle_df[struggle_df['significant_struggle']].groupby('student_id')\
        .agg(struggle_count=('significant_struggle', 'sum'),
             avg_struggle_deviation=('struggle_deviation', 'mean'))\
        .sort_values('struggle_count', ascending=False)
    results['struggling_learners'] = struggling_learners
    print("\nTop Learners by Significant Struggle Count:")
    display(struggling_learners.head(10))

    # Videos where learners struggle most significantly
    struggle_videos = struggle_df[struggle_df['significant_struggle']].groupby('video_title')\
        .agg(struggle_count=('significant_struggle', 'sum'),
             avg_struggle_deviation=('struggle_deviation', 'mean'))\
        .sort_values('struggle_count', ascending=False)
    results['struggle_videos'] = struggle_videos
    print("\nTop Videos by Significant Struggle Count:")
    display(struggle_videos.head(10))

    # Struggle patterns by course
    if 'course_code' in struggle_df.columns:
        course_struggle = struggle_df.groupby('course_code')\
            .agg(total_views=('student_id', 'size'),
                 significant_struggles=('significant_struggle', 'sum'))
        course_struggle['struggle_rate'] = (course_struggle['significant_struggles'] / course_struggle['total_views'] * 100).round(2)


        course_struggle = course_struggle[course_struggle['total_views'] >= min_course_views]
        course_struggle = course_struggle.sort_values('struggle_rate', ascending=False)
        results['course_struggle'] = course_struggle
        print("\nStruggle Rate by Course (min views =", min_course_views, "):")
        display(course_struggle.head(10))

        # Visualization: Top 10 courses by struggle rate
        fig_course = px.bar(
            course_struggle.head(10).reset_index(),
            x='struggle_rate',
            y='course_code',
            orientation='h',
            title='Top 10 Courses by Struggle Rate (%)',
            labels={'struggle_rate':'Struggle Rate (%)','course_code':'Course Code'}
        )
        fig_course.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5)
        fig_course_path = os.path.join(OUTPUT_DIR, 'course_struggle_rate.png')
        fig_course.write_image(fig_course_path)
        print(f"Course struggle rate plot saved to {fig_course_path}")
        results['fig_course_struggle'] = fig_course

    else:
        results['course_struggle'] = None
        print("\n'course_code' not found, skipping course-level analysis.")

    # Visualization: Distribution of struggle deviations
    fig_dev = px.histogram(
        struggle_df, x='struggle_deviation', nbins=50,
        title='Distribution of Struggle Deviations',
        labels={'struggle_deviation':'Deviation from Video Average (%)'}
    )
    fig_dev_path = os.path.join(OUTPUT_DIR, 'struggle_deviation_dist.png')
    fig_dev.write_image(fig_dev_path)
    print(f"Struggle deviation distribution plot saved to {fig_dev_path}")
    results['fig_struggle_deviation'] = fig_dev

    return results
