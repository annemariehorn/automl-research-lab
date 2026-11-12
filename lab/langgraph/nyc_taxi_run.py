"""Run the custom NYC taxi LangGraph workflow."""

import json

from lab.langgraph.nyc_taxi_graph import build_graph


def main():
    graph = build_graph()

    initial_state = {
        "cycle": 0,
        "max_cycles": 3,
        "target_score": 0.99,
        "history": [],
    }

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": 25},
    )

    print("\n" + "=" * 45)
    print("NYC TAXI LANGGRAPH RESULTS")
    print("=" * 45)

    summary = {
        "rows": result["row_count"],
        "cycles": result["cycle"],
        "best_model": result["best_model"],
        "best_validation_f1": result["best_f1_score"],
        "test_accuracy": result["test_accuracy"],
        "test_f1": result["test_f1_score"],
        "stop_reason": result["stop_reason"],
        "experiment_history": result["history"],
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
