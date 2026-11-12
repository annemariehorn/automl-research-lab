from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from airflow.decorators import dag, task

# --------------------------------------------------
# File paths inside the Airflow Docker container
# --------------------------------------------------

REPO = Path("/repo")
DATA_DIR = REPO / "datasets" / "nyc_taxi"

TRIPS = DATA_DIR / "yellow_taxi_sample.parquet"
ZONES = DATA_DIR / "taxi_zone_lookup.csv"

OUTPUT_DIR = REPO / "artifacts" / "nyc_taxi"


DEFAULT_ARGS = {
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}


# --------------------------------------------------
# Define our Airflow DAG
# --------------------------------------------------


@dag(
    dag_id="nyc_taxi_pipeline",
    description="Clean, merge, and store NYC taxi trip data",
    start_date=datetime(2026, 11, 1),
    schedule="@daily",
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["nyc_taxi", "student_project"],
)
def nyc_taxi_pipeline():
    """Our original NYC taxi data engineering pipeline."""

    # ----------------------------------------------
    # Task 1: Clean taxi trip records
    # ----------------------------------------------

    @task
    def clean_taxi_trips():
        """Clean taxi records and create useful features."""

        df = pd.read_parquet(TRIPS)
        original_rows = len(df)

        # Remove trips with invalid distances or fares
        df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)].copy()

        # Treat invalid passenger counts as missing
        df.loc[df["passenger_count"] <= 0, "passenger_count"] = None

        # Fill missing passenger counts with the median
        df["passenger_count"] = df["passenger_count"].fillna(
            df["passenger_count"].median()
        )

        # Extract useful time features
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

        df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek

        # Save cleaned data
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = OUTPUT_DIR / "cleaned_trips.parquet"
        df.to_parquet(output_path, index=False)

        print(f"Original rows: {original_rows}")
        print(f"Cleaned rows: {len(df)}")

        return str(output_path)

    # ----------------------------------------------
    # Task 2: Clean taxi zone lookup data
    # ----------------------------------------------

    @task
    def clean_taxi_zones():
        """Validate and clean the taxi zone lookup table."""

        df = pd.read_csv(ZONES)
        original_rows = len(df)

        # Remove records without a location ID
        df = df.dropna(subset=["LocationID"]).copy()

        # Ensure location IDs are unique
        if df["LocationID"].duplicated().any():
            raise ValueError("Duplicate taxi zone IDs found")

        # Fill missing borough and zone names
        df["Borough"] = df["Borough"].fillna("Unknown")
        df["Zone"] = df["Zone"].fillna("Unknown")

        # Standardize text formatting
        df["Borough"] = df["Borough"].str.strip()
        df["Zone"] = df["Zone"].str.strip()

        # Save cleaned data
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = OUTPUT_DIR / "cleaned_zones.csv"
        df.to_csv(output_path, index=False)

        print(f"Original zones: {original_rows}")
        print(f"Cleaned zones: {len(df)}")

        return str(output_path)

    # ----------------------------------------------
    # Task 3: Merge taxi trips with zone information
    # ----------------------------------------------

    @task
    def merge_taxi_data(trips_path: str, zones_path: str):
        """Combine cleaned taxi trips with pickup zone information."""

        trips = pd.read_parquet(trips_path)
        zones = pd.read_csv(zones_path)

        # Join trips to their pickup neighborhoods
        merged = trips.merge(
            zones,
            left_on="PULocationID",
            right_on="LocationID",
            how="left",
            validate="many_to_one",
        )

        # Verify that every trip matched a zone
        if merged["Borough"].isna().any():
            raise ValueError("Some taxi trips have no matching zone")

        if len(merged) != len(trips):
            raise ValueError("Merge unexpectedly changed row count")

        # Save the merged dataset
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = OUTPUT_DIR / "merged_taxi_data.parquet"

        merged.to_parquet(output_path, index=False)

        print(f"Cleaned trips: {len(trips)}")
        print(f"Merged rows: {len(merged)}")
        print(f"Columns: {len(merged.columns)}")

        return str(output_path)

    # ----------------------------------------------
    # Task 4: Load cleaned data into PostgreSQL
    # ----------------------------------------------

    @task
    def load_to_postgres(merged_path: str):
        """Store merged taxi records in PostgreSQL."""

        df = pd.read_parquet(merged_path)

        # Keep only columns needed for machine learning
        columns = [
            "trip_distance",
            "passenger_count",
            "pickup_hour",
            "pickup_day_of_week",
            "PULocationID",
            "Borough",
            "Zone",
            "fare_amount",
        ]

        df = df[columns].copy()

        # Connect to our dedicated NYC Taxi database
        # Environment variables override local Docker defaults
        conn = psycopg2.connect(
            host=os.getenv("TAXI_DB_HOST", "airflow-db"),
            dbname=os.getenv("TAXI_DB_NAME", "nyc_taxi"),
            user=os.getenv("TAXI_DB_USER", "airflow"),
            password=os.getenv("TAXI_DB_PASSWORD", "airflow"),
        )

        try:
            # The transaction commits only if all operations succeed
            with conn:
                with conn.cursor() as cursor:

                    # Recreate the table on each run
                    cursor.execute("""
                        DROP TABLE IF EXISTS taxi_trips_cleaned;

                        CREATE TABLE taxi_trips_cleaned (
                            trip_distance DOUBLE PRECISION,
                            passenger_count DOUBLE PRECISION,
                            pickup_hour INTEGER,
                            pickup_day_of_week INTEGER,
                            pickup_location_id INTEGER,
                            borough TEXT,
                            zone TEXT,
                            fare_amount DOUBLE PRECISION
                        );
                    """)

                    rows = [tuple(row) for row in df.itertuples(index=False, name=None)]

                    execute_values(
                        cursor,
                        """
                        INSERT INTO taxi_trips_cleaned (
                            trip_distance,
                            passenger_count,
                            pickup_hour,
                            pickup_day_of_week,
                            pickup_location_id,
                            borough,
                            zone,
                            fare_amount
                        ) VALUES %s
                        """,
                        rows,
                        page_size=1000,
                    )

        finally:
            conn.close()

        print(f"Loaded {len(df)} records into PostgreSQL")

        return len(df)

    # ----------------------------------------------
    # Task 5: Read data back from PostgreSQL
    # ----------------------------------------------

    @task
    def read_from_postgres(loaded_rows: int):
        """Retrieve cleaned taxi data for machine learning."""

        conn = psycopg2.connect(
            host=os.getenv("TAXI_DB_HOST", "airflow-db"),
            dbname=os.getenv("TAXI_DB_NAME", "nyc_taxi"),
            user=os.getenv("TAXI_DB_USER", "airflow"),
            password=os.getenv("TAXI_DB_PASSWORD", "airflow"),
        )

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        trip_distance,
                        passenger_count,
                        pickup_hour,
                        pickup_day_of_week,
                        pickup_location_id,
                        borough,
                        zone,
                        fare_amount
                    FROM taxi_trips_cleaned
                """)

                columns = [description[0] for description in cursor.description]

                rows = cursor.fetchall()

                df = pd.DataFrame(rows, columns=columns)

        finally:
            conn.close()

        # Verify that all records were retrieved
        if len(df) != loaded_rows:
            raise ValueError(f"Expected {loaded_rows} rows, found {len(df)}")

        # Save the database output for our ML workflow
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = OUTPUT_DIR / "taxi_ml_ready.parquet"

        df.to_parquet(output_path, index=False)

        print(f"Retrieved {len(df)} rows from PostgreSQL")

        return str(output_path)

    # ----------------------------------------------
    # Register tasks and define dependencies
    # ----------------------------------------------

    # These two tasks can run in parallel
    cleaned_trips = clean_taxi_trips()
    cleaned_zones = clean_taxi_zones()

    # Wait for both cleaning tasks before merging
    merged = merge_taxi_data(cleaned_trips, cleaned_zones)

    # Store the merged data in PostgreSQL
    loaded = load_to_postgres(merged)

    # Read the data back for machine learning
    read_from_postgres(loaded)


# Register the DAG with Airflow
nyc_taxi_pipeline()
