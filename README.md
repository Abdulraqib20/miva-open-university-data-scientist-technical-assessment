# 🔍 Data Scientist Assessment (Video Engagement Data)

## Overview

This technical assessment, as part of the recruitment process for the Data Scientist Position at Miva Open University analyzes Learning Management System (LMS) video engagement data to derive insights into learner behavior, identify areas for improvement, and build predictive models. It is designed to evaluate abilities across key data science domains such as exploratory data analysis, predictive modeling, engagement metrics, clustering, and more.


The technical assessment analysis covers:
1.  **EDA:** Loading data, calculating video engagement stats (mean/median completion), and visualizing top videos.
2.  **Learner Engagement Metric:** Creating a custom score based on completion rate and unique videos watched.
3.  **Churn Prediction & API:** Defining video-level churn (<49% completion) and building a Flask API to predict completion using a trained model.
4.  **Clustering Learners:** Grouping learners using K-Means Clustering based on average completion and unique videos watched.
5.  **Anomaly Detection:** Identifying learners with specific viewing patterns (average completion 40-50% but not watching all videos).
6.  **Learning Path Optimization:** Analyzing videos frequently abandoned mid-watch and identifying drop-off patterns related to courses.
7.  **Predictive Modeling:** Building a XGBoost Classifier model to predict if a learner will complete a video (>=95%) based on their historical average completion and the course.
8.  **Refined Struggle Analysis:** Identifying specific student-video interactions where a learner significantly underperformed compared to the average for that video.

## Project Structure

```

|-- api/
|   |-- models/             # Stores saved model artifacts (pipeline, preprocessor)
|   |-- app.py              # Flask application for the prediction API
|-- data/
|   |-- LMS Video Engagement Data.csv  # Project CSV Dataset
|-- output/
|   |-- (Generated Plots)   # Directory where plots/outputs are saved
|   |-- postman             # Postman Inference Screenshots
|-- src/
|   |-- __init__.py
|   |-- assessment.ipynb    # Main Jupyter Notebook file containing the solution to the technical assessment
|-- utils/                  # folder containing helper modules used in the notebook
|   |-- __init__.py
|   |-- anomaly_detection.py # Contains functions for Task 5 (Anomaly Detection)
|   |-- clustering.py       # Contains functions for Task 4 (Clustering Learners)
|   |-- data_processing.py  # Contains functions for Task 3 (Churn Prediction)
|   |-- eda.py              # Contains functions for Task 1 (EDA)
|   |-- engagement.py       # Contains functions for Task 2 (Learner Engagement Metric)
|   |-- path_optimization.py # Contains functions for Task 6 (Learning Path Optimization)
|   |-- prediction_xgb.py       # Contains functions for Task 7 (Predictive Model Training)
|   |-- struggle_analysis.py # Contains functions for Task 8 (Refined Struggle)
|-- .gitignore
|-- README.md
|-- requirements.txt    # Python dependencies for the project
```

*   `api/`: Contains the Flask API code and necessary files.
*   `data/`: contains the input CSV dataset.
*   `output/`: Stores plots and other output files generated during analysis.
*   `src/`: Contains the main Jupyter Notebook which I used to run the analysis.
*   `utils/`: Contains helper Python modules, broken down by task, to keep the notebook clean and code reusable.

## Setup and Installation

1.  **Prerequisites:**
    *   Python (version 3.9 or higher recommended)
    *   `pip` (Python package installer)
2.  **Clone Repository:**
    ```bash
    git clone https://github.com/Abdulraqib20/miva-open-university-data-scientist-technical-assessment
    cd miva-open-university-data-scientist-technical-assessment
    ```
3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
4.  **Install Dependencies:**
    *   **For the main analysis notebook:**
        ```bash
        pip install -r requirements.txt
        ```
5.  **Add Data:** Place the `LMS Video Engagement Data.csv` file into the `data/` directory.

## Running the Notebook

1.  **Launch Jupyter:** Ensure the virtual environment is active and run:
    ```bash
    jupyter notebook
    ```
2.  **Open Notebook:** Navigate to and open `src/assessment.ipynb` in the Jupyter interface.
3.  **Run Cells:** Execute the cells sequentially. The notebook is structured to follow the assessment tasks.
    *   The notebook uses `autoreload` to automatically pick up changes made in the `utils/` scripts without restarting the kernel (after the initial import).
    *   Outputs, including tables, statistics, and interactive Plotly visualizations, will be displayed directly in the notebook.
    *   Some static plots are saved as image files in the `output/` directory.

## Running the Prediction API

1.  **Train the Model:** Ensure you have run the relevant cells in `src/assessment.ipynb` to execute `utils/prediction.py`, which trains the model and saves the necessary files (`completion_predictor.pkl`, etc.) into the `api/models/` directory.
2.  **Navigate to API Directory:** Open a new terminal (with the virtual environment activated):
    ```bash
    cd api
    ```
3.  **Run the Flask App:**
    ```bash
    python app.py
    ```
    The API will start, usually on `http://localhost:5000` or `http://127.0.0.1:5000`. It will indicate if the model pipeline loaded successfully.
4.  **Test the API:** Use curl/Postman to send a `POST` request to the `/predict` endpoint.

    **Example using `curl`:**
    ```bash
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{
               "learner_avg_completion_rate": 75.5,
               "course_code": "FIN201"
             }'
    ```

    **Input JSON Format:**
    ```json
    {
      "learner_avg_completion_rate": <float (0-100)>,
      "course_code": "<string>"
    }
    ```

    **Example Success Response:**
    ```json
    {
      "message": "Prediction successful.",
      "prediction": 1,
      "prediction_label": "Will Complete",
      "probability_complete": 0.856,
      "probability_not_complete": 0.144
    }
    ```

    **Example Error Response (Invalid Input):**
    ```json
    {
      "error": "Invalid value for 'learner_avg_completion_rate'. Must be between 0 and 100 (inclusive)."
    }
    ```

## 📬 Prediction API Snapshots

<p align="center">
  <img src="output/postman/Screenshot%202025-05-01%20193241.png" alt="Postman Screenshot 1" width="600" style="margin-bottom:20px;" />
</p>
<p align="center">
  <img src="output/postman/Screenshot%202025-05-01%20193358.png" alt="Postman Screenshot 2" width="600" style="margin-bottom:20px;" />
</p>
<p align="center">
  <img src="output/postman/Screenshot%202025-05-01%20193717.png" alt="Postman Screenshot 3" width="600" style="margin-bottom:20px;" />
</p>
<p align="center">
  <img src="output/postman/Screenshot%202025-05-01%20193813.png" alt="Postman Screenshot 4" width="600" style="margin-bottom:20px;" />
</p>
<p align="center">
  <img src="output/postman/Screenshot%202025-05-01%20193921.png" alt="Postman Screenshot 5" width="600" style="margin-bottom:20px;" />
</p>

## Utility Scripts (`utils/`)

*   `eda.py`: Handles initial data loading, calculation of basic video stats (average/median completion), and plotting top videos (using Plotly).
*   `engagement.py`: Calculates the custom learner engagement score based on normalized average completion and unique videos watched.
*   `data_processing.py`: Contains functions for general data transformations, such as adding the binary churn label (<49% completion).
*   `clustering.py`: Implements learner clustering using K-Means based on engagement features (average completion, unique videos). Includes functions for finding optimal K (using silhouette analysis) and visualizing clusters.
*   `anomaly_detection.py`: Identifies learners exhibiting specific anomalous behavior (average completion 40-50% but not watching all videos).
*   `path_optimization.py`: Analyzes learning paths by identifying frequently abandoned videos and analyzing drop-off patterns, including course-specific breakdowns and visualizations.
*   `prediction.py`: Contains the end-to-end workflow for the predictive modeling task: feature engineering, training an XGBoost model (including preprocessing), evaluation, and saving the model artifacts for the API.
*   `struggle_analysis.py`: Implements the custom Task 8. Calculates video-specific struggle scores based on deviation from average completion and analyzes patterns of significant struggle.

## Outputs

*   Key statistics and tables are printed within the `src/assessment.ipynb` notebook.
*   Interactive plots (EDA, Clustering, Path Optimization, Struggle Analysis) are displayed in the notebook.
*   Static analysis plots (e.g., silhouette scores) and specific pattern plots are saved to the `output/` directory.
*   The trained predictive model and preprocessor are saved in `api/models/`.
*   The prediction API provides real-time completion predictions via the `/predict` endpoint.

