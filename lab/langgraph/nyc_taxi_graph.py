"""Custom LangGraph workflow for NYC taxi fare classification."""

from pathlib import Path
from tempfile import mkdtemp
from typing import TypedDict

import pandas as pd

from langgraph.graph import StateGraph, END

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------
# PATHS
# --------------------------------------------------

REPO = Path(__file__).resolve().parents[2]

DATA_PATH = REPO / "artifacts" / "nyc_taxi" / "taxi_ml_ready.parquet"


# --------------------------------------------------
# GRAPH STATE
# --------------------------------------------------


class TaxiState(TypedDict, total=False):
    split_dir: str
    hypothesis: dict
    context: dict
    feedback: str
    recommendation: str
    current_experiment: dict
    dataset_path: str
    row_count: int
    feature_columns: list[str]

    model_name: str
    accuracy: float
    f1_score: float

    cycle: int
    max_cycles: int
    target_score: float

    history: list[dict]

    best_model: str
    best_f1_score: float

    decision: str
    stop_reason: str

    test_accuracy: float
    test_f1_score: float


# --------------------------------------------------
# NODE 1: LOAD DATA
# --------------------------------------------------


def load_data(state: TaxiState) -> dict:
    """Load and validate the dataset produced by Airflow."""

    path = Path(state.get("dataset_path", str(DATA_PATH)))

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run the Airflow DAG first."
        )

    if state.get("max_cycles", 3) < 1:
        raise ValueError("max_cycles must be at least 1")

    df = pd.read_parquet(path)

    required_columns = [
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "pickup_day_of_week",
        "borough",
        "fare_amount",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df)} cleaned taxi trips")

    return {
        "dataset_path": str(path),
        "row_count": len(df),
        "feature_columns": required_columns[:-1],
        "cycle": 0,
        "history": [],
    }


# --------------------------------------------------
# NODE 2: PREPARE DATA
# --------------------------------------------------


def prepare_data(state: TaxiState) -> dict:
    """Create train, validation, and test splits."""

    df = pd.read_parquet(state["dataset_path"]).copy()

    df = df.dropna(
        subset=[
            "trip_distance",
            "passenger_count",
            "pickup_hour",
            "pickup_day_of_week",
            "borough",
            "fare_amount",
        ]
    )

    # 60% training, 40% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        random_state=42,
    )

    # 20% validation, 20% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
    )

    # Determine target threshold using training data only
    median_fare = train_df["fare_amount"].median()

    for split in (train_df, val_df, test_df):
        split["high_fare"] = (split["fare_amount"] > median_fare).astype(int)

    # Never overwrite existing datasets or share mutable splits across runs.
    split_dir = Path(mkdtemp(prefix="nyc-taxi-splits-"))

    train_df.to_parquet(
        split_dir / "train.parquet",
        index=False,
    )

    val_df.to_parquet(
        split_dir / "validation.parquet",
        index=False,
    )

    test_df.to_parquet(
        split_dir / "test.parquet",
        index=False,
    )

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"High-fare threshold: ${median_fare:.2f}")

    return {
        "split_dir": str(split_dir),
        "feature_columns": [
            "trip_distance",
            "passenger_count",
            "pickup_hour",
            "pickup_day_of_week",
            "borough",
        ],
    }


# --------------------------------------------------
# SHARED MODEL DEFINITIONS
# --------------------------------------------------


def make_preprocessor():
    """Build preprocessing fitted only on model-training data."""

    numeric_features = [
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "pickup_day_of_week",
    ]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["borough"],
            ),
        ]
    )


def get_models():
    """Candidate models for our custom experiment loop."""

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        ),
    }


# --------------------------------------------------
# NODE 3: TRAIN MODEL
# --------------------------------------------------


def train_model(state: TaxiState) -> dict:
    """Train and validate the next candidate model."""

    train_df = pd.read_parquet(Path(state["split_dir"]) / "train.parquet")

    val_df = pd.read_parquet(Path(state["split_dir"]) / "validation.parquet")

    features = state["feature_columns"]

    X_train = train_df[features]
    y_train = train_df["high_fare"]

    X_val = val_df[features]
    y_val = val_df["high_fare"]

    models = get_models()
    model_name = state["hypothesis"]["model"]
    cycle = state.get("cycle", 0)

    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("classifier", models[model_name]),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    experiment = {
        "cycle": cycle + 1,
        "model": model_name,
        "configuration": models[model_name].get_params(),
        "hypothesis": state["hypothesis"]["hypothesis"],
        "rationale": state["hypothesis"]["rationale"],
        "accuracy": float(accuracy),
        "f1_score": float(f1),
    }

    print(f"\nCycle {cycle + 1}: {model_name}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation F1-score: {f1:.4f}")

    return {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "cycle": cycle + 1,
        "current_experiment": experiment,
    }


# --------------------------------------------------
# NODE 4: EVALUATE MODEL
# --------------------------------------------------


def evaluate_model(state: TaxiState) -> dict:
    """Decide whether to try another model."""

    current = state["current_experiment"]
    history = state.get("history", []) + [current]

    best = max(
        history,
        key=lambda result: result["f1_score"],
    )

    target_score = state.get("target_score", 0.85)
    max_cycles = state.get("max_cycles", 3)

    if current["f1_score"] >= target_score:
        decision = "stop"
        reason = "Target validation F1-score achieved"

    elif state["cycle"] >= max_cycles:
        decision = "stop"
        reason = "Maximum training cycles reached"

    elif len(history) >= len(get_models()):
        decision = "stop"
        reason = "All distinct candidate configurations evaluated"

    else:
        decision = "retry"
        reason = "Validation F1 below target; request a new hypothesis"

    previous_best = max((r["f1_score"] for r in history[:-1]), default=0.0)
    improvement = current["f1_score"] - previous_best
    # Strong linear evidence favors an ensemble; weaker evidence first tests a
    # smaller nonlinear tree; then test the remaining untried family.
    if current["model"] == "Logistic Regression":
        recommendation = "Random Forest" if current["f1_score"] >= 0.88 else "Decision Tree"
    elif current["model"] == "Decision Tree":
        recommendation = "Random Forest"
    else:
        recommendation = "Decision Tree"
    remaining = [name for name in get_models()
                 if name not in {r["model"] for r in history}]
    if recommendation not in remaining:
        recommendation = remaining[0] if remaining else "none (candidates exhausted)"
    next_direction = recommendation if decision == "retry" else "none (model selection finished)"
    feedback = (
        f"{current['model']} validation F1={current['f1_score']:.6f}; "
        f"change versus prior best={improvement:+.6f}; "
        f"best={best['model']} ({best['f1_score']:.6f}), target={target_score:.4f}. "
        f"{reason}. Next untried direction: {next_direction}."
    )

    print(f"Decision: {decision}")
    print(f"Reason: {reason}")
    print(f"Best model so far: {best['model']}")

    return {
        "feedback": feedback,
        "recommendation": recommendation,
        "best_model": best["model"],
        "best_f1_score": best["f1_score"],
        "decision": decision,
        "stop_reason": reason,
    }


def memory_writer(state: TaxiState) -> dict:
    """File the completed trial in the single canonical history (idempotently)."""
    history = list(state.get("history", []))
    experiment = {**state["current_experiment"],
                  "feedback": state["feedback"], "decision": state["decision"]}
    if any(r["cycle"] == experiment["cycle"] for r in history):
        return {"history": history}
    return {"history": history + [experiment]}


def context_builder(state: TaxiState) -> dict:
    """Compact validation-only brief; do not copy the complete experiment log."""
    history = state["history"]
    best = max(history, key=lambda r: r["f1_score"])
    return {"context": {
        "tried_models": [r["model"] for r in history],
        "summary": "; ".join(f"{r['model']}: F1={r['f1_score']:.6f}" for r in history),
        "best_model": best["model"],
        "best_f1_score": best["f1_score"],
        "feedback": state["feedback"],
        "recommendation": state["recommendation"],
    }}


def hypothesis_generator(state: TaxiState) -> dict:
    """Choose an untried experiment using the judge's validation-only brief."""
    context = state.get("context", {})
    tried = context.get("tried_models", [])
    available = [name for name in get_models() if name not in tried]
    if not available:
        raise ValueError("No untried candidate remains")
    if not tried:
        model = "Logistic Regression"
        hypothesis = "A linear classifier establishes a distance/time/borough baseline."
        rationale = "First experiment; no validation evidence yet."
    else:
        recommended = context["recommendation"]
        model = recommended if recommended in available else available[0]
        hypothesis = (
            "An ensemble may improve generalization through averaging nonlinear trees."
            if model == "Random Forest" else
            "A shallow tree may capture nonlinear fare boundaries with limited complexity."
        )
        rationale = f"{context['summary']}. Judge: {context['feedback']} Testing untried {model}."
    return {"hypothesis": {"model": model, "hypothesis": hypothesis, "rationale": rationale}}


# --------------------------------------------------
# CONDITIONAL ROUTER
# --------------------------------------------------


def route_after_evaluation(state: TaxiState) -> str:
    """After recording the trial, route on the judge's unchanged decision."""

    if state["decision"] == "retry":
        return "retry"

    return "final_test"


# --------------------------------------------------
# NODE 5: FINAL HOLDOUT TEST
# --------------------------------------------------


def final_test(state: TaxiState) -> dict:
    """Test the best model on untouched holdout data."""

    train_df = pd.read_parquet(Path(state["split_dir"]) / "train.parquet")

    val_df = pd.read_parquet(Path(state["split_dir"]) / "validation.parquet")

    test_df = pd.read_parquet(Path(state["split_dir"]) / "test.parquet")

    development_df = pd.concat(
        [train_df, val_df],
        ignore_index=True,
    )

    features = state["feature_columns"]

    X_train = development_df[features]
    y_train = development_df["high_fare"]

    X_test = test_df[features]
    y_test = test_df["high_fare"]

    best_name = state["best_model"]

    models = get_models()

    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("classifier", models[best_name]),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print("\nFINAL HOLDOUT EVALUATION")
    print(f"Best model: {best_name}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test F1-score: {f1:.4f}")

    return {
        "test_accuracy": float(accuracy),
        "test_f1_score": float(f1),
    }


# --------------------------------------------------
# BUILD LANGGRAPH
# --------------------------------------------------


def build_graph():
    """Connect all nodes into our custom workflow."""

    graph = StateGraph(TaxiState)

    graph.add_node("load_data", load_data)
    graph.add_node("prepare_data", prepare_data)
    graph.add_node("hypothesis_generator", hypothesis_generator)
    graph.add_node("memory_writer", memory_writer)
    graph.add_node("context_builder", context_builder)
    graph.add_node("train_model", train_model)
    graph.add_node("evaluate_model", evaluate_model)
    graph.add_node("final_test", final_test)

    graph.set_entry_point("load_data")

    graph.add_edge(
        "load_data",
        "prepare_data",
    )

    graph.add_edge(
        "prepare_data",
        "hypothesis_generator",
    )

    graph.add_edge(
        "train_model",
        "evaluate_model",
    )

    graph.add_edge("evaluate_model", "memory_writer")
    graph.add_conditional_edges(
        "memory_writer",
        route_after_evaluation,
        {
            "retry": "context_builder",
            "final_test": "final_test",
        },
    )

    graph.add_edge("hypothesis_generator", "train_model")
    graph.add_edge("context_builder", "hypothesis_generator")
    graph.add_edge("final_test", END)

    return graph.compile()


def get_status():
    """Describe the NYC Taxi graph to the live interface."""
    compiled = build_graph().get_graph()
    return {
        "ready": True,
        "nodes": [name for name in compiled.nodes
                  if name not in {"__start__", "__end__"}],
        "edges": [[edge.source, edge.target]
                  for edge in compiled.edges
                  if edge.source != "__start__" and edge.target != "__end__"],
        "missing_nodes": [],
        "error": None,
    }


def stream_lab_loop(
    dataset_path=None,
    target_column=None,
    target_score=0.99,
    max_cycles=3,
    seed=42,
    model=None,
):
    """Stream NYC Taxi LangGraph execution to the live interface."""
    if max_cycles < 1:
        raise ValueError("max_cycles must be at least 1")
    graph = build_graph()

    state = {
        "cycle": 0,
        "max_cycles": max_cycles,
        "target_score": target_score,
        "history": [],
    }

    if dataset_path:
        state["dataset_path"] = str(dataset_path)

    yield {"type": "run_started"}

    previous_node = None
    # Debug task events arrive before execution, unlike update-only events.
    for event in graph.stream(
        state,
        stream_mode="debug",
        config={"recursion_limit": 8 + max_cycles * 6},
    ):
        payload = event["payload"]
        if event["type"] == "task":
            yield {"type": "node_started", "node": payload["name"],
                   "previous_node": previous_node}
        elif event["type"] == "task_result":
            if payload.get("error"):
                raise RuntimeError(payload["error"])
            node_output = dict(payload["result"])
            state.update(node_output)
            previous_node = payload["name"]
            yield {
                "type": "node_completed",
                "node": payload["name"],
                "output": node_output,
            }

    yield {
        "type": "run_completed",
        "result": state,
    }
