# Healthcare Device Data Anomaly Detection

This project focuses on detecting anomalies in physiological and motion data collected from wearable devices using the **PPG-DaLiA dataset**. The pipeline preprocesses raw data, extracts meaningful features, and applies machine learning models for anomaly detection. The primary algorithm used is the **Isolation Forest**, which is well-suited for unsupervised anomaly detection.


## Key Files in the Pipeline

### 1. **[notebooks/preprocess.ipynb](notebooks/preprocess.ipynb)**

This notebook is the cornerstone of the preprocessing pipeline. It handles the following tasks:

- **Data Loading**: Reads raw physiological and motion data from `.pkl` files in the `data/PPG_FieldStudy` directory.
- **Signal Segmentation**: Segments raw signals (BVP, ACC, TEMP) into 8-second windows with a 2-second overlap.
- **Activity Data Integration**: Maps activity logs to activity IDs and merges them with the segmented signals.
- **Feature Extraction**: Extracts 22 meaningful features, including:
  - **BVP Features**: Mean, standard deviation, kurtosis, frequency components, etc.
  - **ACC Features**: Mean, standard deviation, magnitude, etc.
  - **TEMP Features**: Mean, standard deviation, range.
  - **Contextual Features**: Activity labels and heart rate (BPM).
- **Output**: Saves the processed data for each patient as `.pkl` files in the `processed_data/` directory.

This notebook ensures that the raw data is transformed into a structured format, ready for anomaly detection.

### 2. **[model/anomaly_model.ipynb](model/anomaly_model.ipynb)**

This notebook applies the **Isolation Forest** algorithm to detect anomalies in the preprocessed data. It includes:

- **Data Loading**: Loads the processed `.pkl` files from the `processed_data/` directory.
- **Anomaly Detection**: Uses the Isolation Forest algorithm to identify anomalies in the data.
- **Result Analysis and Evaluation**:
  - **Anomaly Distribution**: Analyzes the distribution of anomalies across patients and activities.
  - **Temporal Analysis**: Examines anomalies over time to identify patterns.
  - **SHAP Feature Importance**: Explains the model's predictions by analyzing feature importance using SHAP (SHapley Additive exPlanations).
  - **Visualization**: Includes bar plots, heatmaps, and other visualizations to interpret the results.

This notebook provides insights into the detected anomalies and evaluates the model's performance.

## Pipeline Overview

1. **Preprocessing**:
   - Run `notebooks/preprocess.ipynb` to preprocess the raw data and extract features.
   - The output is saved as `.pkl` files in the `processed_data/` directory.

2. **Anomaly Detection**:
   - Run `model/anomaly_model.ipynb` to detect anomalies using the Isolation Forest algorithm.
   - Analyze the results, including anomaly distribution, temporal patterns, and feature importance.

## Dependencies

The project requires the following Python libraries:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `shap`

Install the dependencies using the following command:

```bash
pip install -r [requirements.txt](http://_vscodecontentref_/3)

## How to Run

1. Clone the repository and navigate to the project directory.
2. Ensure the `data/PPG_FieldStudy` directory contains the raw `.pkl` files.
3. Run the preprocessing notebook:
   ```bash
   jupyter notebook notebooks/preprocess.ipynb
4. Run the anomaly detection notebook:
   ```bash
   jupyter notebook model/anomaly_model.ipynb

   ## Results

The results of the anomaly detection process, including visualizations and feature importance analysis, are saved in the `visualizations/` directory. These insights can be used to understand the behavior of wearable devices and identify potential issues.

## Future Work

- Extend the pipeline to include additional anomaly detection algorithms.
- Incorporate more advanced feature engineering techniques.
- Explore supervised learning approaches if labeled anomaly data becomes available.