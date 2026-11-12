
# NYC Taxi Data Pipeline and ML Workflow

## Project Overview

This project uses Apache Airflow and LangGraph to build an automated data processing and machine learning workflow using New York City yellow taxi trip data.

The objective is to predict whether a taxi fare is above the median fare using trip characteristics such as distance, passenger count, pickup time, and pickup borough.

The project consists of two parts:
1. An Airflow DAG that cleans, merges, and stores taxi data in PostgreSQL.
2. A custom LangGraph workflow that trains and evaluates classification models using conditional looping logic.

## Part 1: Apache Airflow Pipeline

### Datasets

Two datasets from the NYC Taxi & Limousine Commission (TLC) are used:

- January 2025 Yellow Taxi Trip Records
- NYC Taxi Zone Lookup Table

The original trip dataset contains approximately 3.48 million records. A reproducible sample of 20,000 trips was selected using `random_state=42`.

### DAG Structure

The custom DAG is located at:

`lab/airflow/dags/nyc_taxi_pipeline.py`

It contains five tasks:

1. **clean_taxi_trips:** Removes trips with nonpositive distances or fares, handles missing passenger counts, and creates pickup hour and weekday features.
2. **clean_taxi_zones:** Validates location IDs and cleans borough and zone information.
3. **merge_taxi_data:** Joins taxi trips with zone information using pickup location IDs.
4. **load_to_postgres:** Loads the merged dataset into the PostgreSQL `taxi_trips_cleaned` table.
5. **read_from_postgres:** Reads the cleaned records from PostgreSQL and saves a machine-learning-ready Parquet file.

The two cleaning tasks run independently before the merge.

### Scheduling

The DAG uses the following Airflow settings:

- Schedule: `@daily`
- Start date: November 1, 2026
- Catchup: `False`
- Maximum active runs: 1

Each scheduled execution reprocesses the same January 2025 sample. It does not automatically download new daily taxi records.

### Data Pipeline Results

- Initial sample: 20,000 trips
- Cleaned dataset: 18,689 trips
- Records removed: 1,311
- Average fare after cleaning: $17.92

The cleaned data is stored in PostgreSQL and exported to:

`artifacts/nyc_taxi/taxi_ml_ready.parquet`

## Part 2: Custom LangGraph ML Workflow

### Objective

Predict whether a taxi trip has a fare above the training-set median of $12.80.

The prediction target is `high_fare`, where:

- `1`: Fare greater than $12.80
- `0`: Fare less than or equal to $12.80

The median threshold is calculated using training data only to avoid test-data leakage.

### Input Features

The classification models use:

- Trip distance
- Passenger count
- Pickup hour
- Pickup day of week
- Pickup borough

The actual fare amount is excluded from model inputs because it defines the target.

### Graph Structure

The custom graph is implemented in:

`lab/langgraph/nyc_taxi_graph.py`

It contains five nodes:

1. **load_data:** Loads and validates the Airflow output.
2. **prepare_data:** Creates training, validation, and test splits.
3. **train_model:** Trains a candidate classification model.
4. **evaluate_model:** Evaluates validation performance and decides whether to retry.
5. **final_test:** Retrains the best candidate on development data and evaluates it on the held-out test set.

### Conditional Loop

After each training cycle, the workflow checks the validation F1-score.

- If the score reaches the target, the workflow stops experimenting.
- Otherwise, it tries another model, provided the maximum number of cycles has not been reached.
- After stopping, it selects the candidate with the highest validation F1-score.

The workflow compares Logistic Regression, Decision Tree, and Random Forest.

This is a deterministic, rule-based LangGraph workflow rather than an LLM-powered agent.

### Data Splitting

- Training: 60% (11,213 records)
- Validation: 20% (3,738 records)
- Test: 20% (3,738 records)

The final test dataset is reserved for evaluation after model selection.

### Model Results

To demonstrate all three training cycles, the validation F1 target was set to 0.99 with a maximum of three cycles.

| Model | Validation Accuracy | Validation F1 |
| --- | --- | --- |
| Logistic Regression | 0.8965 | 0.8864 |
| Decision Tree | 0.9021 | 0.8922 |
| Random Forest | 0.9016 | 0.8924 |

**Selected model:** Random Forest

**Final holdout results:**

- Test accuracy: 0.9026 (90.26%)
- Test F1-score: 0.8963

The workflow stopped after three cycles because it reached the maximum number of allowed attempts.

## Running the Workflow

### Airflow

Start the Docker-based Airflow environment from the repository root:

```bash
make airflow
```

Open Airflow at `http://localhost:8088` and trigger the `nyc_taxi_pipeline` DAG.

### LangGraph

From the repository root, activate the Python environment with the required dependencies installed and run:

```bash
python -m lab.langgraph.nyc_taxi_run
```

The command prints the training cycles, validation scores, routing decisions, selected model, and final test metrics.

## Screenshots

### Airflow DAG

![Successful Airflow pipeline](../../screenshots/nyc_taxi_airflow_success.png)

### LangGraph Structure

![Custom LangGraph workflow](../../screenshots/nyc_taxi_langgraph_graph.png)

### LangGraph Training Cycles

![LangGraph training cycles](../../screenshots/nyc_taxi_langgraph_cycles.png)

### Final Evaluation

![LangGraph evaluation results](../../screenshots/nyc_taxi_langgraph_results.png)

## Conclusion

This project demonstrates a reproducible workflow connecting data engineering and machine learning.

Airflow orchestrates data cleaning, merging, PostgreSQL storage, and data retrieval. LangGraph uses the resulting dataset to compare classification models, repeat experiments based on validation performance, and evaluate the selected model on held-out data.

The final Random Forest classifier achieved 90.26% test accuracy and an F1-score of 0.8963.
