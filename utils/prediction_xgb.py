import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='white', palette='muted')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import xgboost as xgb

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'api', 'models')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

################################ FEATURE ENGINEERING ################################
def create_prediction_features(df, completion_threshold=95):
    """
    Engineers features for predicting video completion.

    Features:
    - learner_avg_completion_rate: Learner's average completion rate across all videos watched.
    - course_code: The course the video belongs to.

    Target:
    - will_complete: 1 if completion_rate_percent >= threshold, 0 otherwise.

    Args:
        df (pd.DataFrame): DataFrame with student_id, course_code, completion_rate_percent.
        completion_threshold (int): Percentage threshold to define 'completion'.

    Returns:
        pd.DataFrame: DataFrame with features and target, or empty DataFrame on error.
    """
    required_cols = ['student_id', 'course_code', 'completion_rate_percent']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input DataFrame must contain {required_cols}.")
        return pd.DataFrame()

    if df.empty:
        print("Input DataFrame is empty.")
        return pd.DataFrame()

    print("Engineering New Features...")
    # Calculate learner's overall average completion rate
    learner_avg = df.groupby('student_id')['completion_rate_percent'].transform('mean')
    df['learner_avg_completion_rate'] = learner_avg

    # target variable
    df['will_complete'] = (df['completion_rate_percent'] >= completion_threshold).astype(int)

    # Select final features and target
    features_df = df[['learner_avg_completion_rate', 'course_code', 'will_complete']].copy()

    # i dropped rows with NaN in features (if a student only has one record)
    features_df = features_df.dropna(subset=['learner_avg_completion_rate', 'course_code'])

    print(f"Feature engineering complete. Shape: {features_df.shape}")
    print(f"Target distribution (will_complete) Value Counts:\n{features_df['will_complete'].value_counts()}")

    return features_df


################################ MODEL TRAINING AND EVALUATION ################################
def train_completion_model(features_df, model_filename='completion_predictor.pkl', preprocessor_filename='completion_preprocessor.pkl'):
    """
    Trains a model to predict video completion.

    Args:
        features_df (pd.DataFrame): DataFrame from create_prediction_features.
        model_filename (str): Filename to save the trained pipeline.
        preprocessor_filename (str): Filename to save the fitted preprocessor.

    Returns:
        tuple: (trained_pipeline, evaluation_metrics) or (None, None) on error.
    """
    if features_df.empty or 'will_complete' not in features_df.columns:
        print("Error: Input DataFrame is empty or missing target column.")
        return None, None

    print("Starting XGBoost model training...")
    X = features_df.drop(['will_complete'], axis=1)
    y = features_df['will_complete']

    # Define numeric and categorical features
    numeric_features = ['learner_avg_completion_rate']
    categorical_features = ['course_code']

    # Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the full pipeline with preprocessing and XGBoost model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            objective='binary:logistic'
        ))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Train the model
    print("Fitting model pipeline...")
    try:
        model_pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model fitting: {e}")
        return None, None

    # Evaluate the model
    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\nModel Evaluation Results:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"Recall Score: {recall:.4f}")
    print(f"Precision Score: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    conf_matrix_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    labels = np.array([
        [f"{conf_matrix[i, j]}\n({conf_matrix_pct[i, j]*100:.1f}%)" for j in range(conf_matrix.shape[1])]
        for i in range(conf_matrix.shape[0])
    ])

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_pct, annot=labels, fmt='', cmap='Blues', cbar_kws={'format': '%.0f%%'})
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.yticks([0.5, 1.5], ['Not Complete', 'Will Complete'])
    plt.xticks([0.5, 1.5], ['Not Complete', 'Will Complete'])
    plt.tight_layout()
    plt.show();

    evaluation_metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1 score': f1,
        'roc_auc': roc_auc,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

    # Save the preprocessor and the full pipeline
    preprocessor_path = os.path.join(MODEL_DIR, preprocessor_filename)
    model_path = os.path.join(MODEL_DIR, model_filename)

    try:
        preprocessor.fit(X)
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")

        # Save the trained pipeline
        joblib.dump(model_pipeline, model_path)
        print(f"Trained model pipeline saved to {model_path}")

        # Save feature names used by the preprocessor
        # Get feature names after one-hot encoding
        feature_names = preprocessor.get_feature_names_out()
        feature_names_path = os.path.join(MODEL_DIR, 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"Feature names saved to {feature_names_path}")

    except Exception as e:
        print(f"Error saving model/preprocessor: {e}")
        return None, evaluation_metrics

    print("Model training and saving complete.")
    return model_pipeline, evaluation_metrics
