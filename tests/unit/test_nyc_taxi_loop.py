"""Part 2 contracts, using the existing Airflow export without changing it.

Run: python -m unittest discover -s tests/unit -p test_nyc_taxi_loop.py
"""
import hashlib
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lab.langgraph import nyc_taxi_graph as taxi


class TaxiLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_hash = hashlib.sha256(taxi.DATA_PATH.read_bytes()).hexdigest()
        cls.events = list(taxi.stream_lab_loop(target_score=0.99, max_cycles=3))
        cls.result = cls.events[-1]["result"]

    def test_graph_and_stream(self):
        completed = [e["node"] for e in self.events if e["type"] == "node_completed"]
        self.assertEqual(completed, ["load_data", "prepare_data", "hypothesis_generator",
            "train_model", "evaluate_model", "memory_writer", "context_builder",
            "hypothesis_generator", "train_model", "evaluate_model", "memory_writer",
            "context_builder", "hypothesis_generator", "train_model", "evaluate_model", "memory_writer", "final_test"])
        previous = None
        for event in self.events:
            if event["type"] == "node_started":
                self.assertEqual(event["previous_node"], previous)
            elif event["type"] == "node_completed":
                previous = event["node"]
        started = [e["node"] for e in self.events if e["type"] == "node_started"]
        self.assertEqual(started, completed)
        for index, event in enumerate(self.events):
            if event["type"] == "node_started":
                self.assertEqual(self.events[index + 1]["type"], "node_completed")
        self.assertEqual(set(completed), set(taxi.get_status()["nodes"]))
        self.assertEqual(completed.count("final_test"), 1)

    def test_selection_memory_and_feedback(self):
        history = self.result["history"]
        self.assertEqual(len(history), 3)
        self.assertEqual(len({r["model"] for r in history}), 3)
        best = max(history, key=lambda r: r["f1_score"])
        self.assertEqual(self.result["best_model"], best["model"])
        self.assertEqual(self.result["best_f1_score"], best["f1_score"])
        for r in history:
            self.assertIn("configuration", r)
            self.assertIn("feedback", r)
        self.assertIn(history[0]["feedback"], history[1]["rationale"])
        self.assertEqual(taxi.memory_writer(self.result)["history"], history)

    def test_judge_leaves_recording_to_memory_writer(self):
        state = {"history": [], "current_experiment": {"cycle": 1,
                 "model": "Logistic Regression", "configuration": {"max_iter": 1000},
                 "accuracy": .9, "f1_score": .88}, "cycle": 1, "target_score": .8}
        verdict = taxi.evaluate_model(state)
        self.assertNotIn("history", verdict)
        self.assertEqual(state["history"], [])
        state.update(verdict)
        state.update(taxi.memory_writer(state))
        self.assertEqual(len(state["history"]), 1)
        self.assertEqual(state["history"][0]["feedback"], verdict["feedback"])
        self.assertEqual(taxi.memory_writer(state)["history"], state["history"])

    def test_full_precision_breaks_display_tie(self):
        lower, higher = .892163, .892200
        self.assertEqual(f"{lower:.4f}", f"{higher:.4f}")
        for prior, current, winner in [(lower, higher, "Random Forest"),
                                      (higher, lower, "Decision Tree")]:
            state = {"history": [{"cycle": 1, "model": "Decision Tree", "f1_score": prior}],
                     "current_experiment": {"cycle": 2, "model": "Random Forest", "f1_score": current},
                     "cycle": 2, "max_cycles": 3, "target_score": .99}
            verdict = taxi.evaluate_model(state)
            self.assertEqual(verdict["best_model"], winner)
            self.assertEqual(verdict["best_f1_score"], higher)

    def test_policy_changes_with_validation_evidence(self):
        def propose(f1):
            state = {"current_experiment": {"cycle": 1, "model": "Logistic Regression",
                     "f1_score": f1}, "cycle": 1, "max_cycles": 3, "target_score": .99}
            state.update(taxi.evaluate_model(state))
            state.update(taxi.memory_writer(state))
            state.update(taxi.context_builder(state))
            # Extraneous holdout values cannot influence the policy.
            state["test_f1_score"] = 1.0
            return taxi.hypothesis_generator(state)["hypothesis"]["model"]
        self.assertEqual(propose(.7), "Decision Tree")
        self.assertEqual(propose(.9), "Random Forest")

    def test_accept_budget_and_candidate_exhaustion(self):
        for target, budget, cycles, reason in [(.8, 3, 1, "Target"),
                (.99, 1, 1, "Maximum"), (.99, 10, 3, "All distinct")]:
            with self.subTest(target=target, budget=budget):
                events = list(taxi.stream_lab_loop(target_score=target, max_cycles=budget))
                result = events[-1]["result"]
                self.assertEqual(result["cycle"], cycles)
                self.assertEqual([r["cycle"] for r in result["history"]], list(range(1, cycles + 1)))
                self.assertTrue(result["stop_reason"].startswith(reason))
                self.assertEqual(sum(e.get("node") == "final_test" and
                    e["type"] == "node_completed" for e in events), 1)
        with self.assertRaises(ValueError):
            list(taxi.stream_lab_loop(max_cycles=0))

    def test_split_and_no_leakage(self):
        root = Path(self.result["split_dir"])
        train, val, test = [pd.read_parquet(root / f"{name}.parquet")
                            for name in ("train", "validation", "test")]
        self.assertEqual([len(train), len(val), len(test)], [11213, 3738, 3738])
        threshold = train.fare_amount.median()
        for split in (train, val, test):
            self.assertTrue((split.high_fare == (split.fare_amount > threshold).astype(int)).all())
        self.assertNotIn("fare_amount", self.result["feature_columns"])
        self.assertNotIn("high_fare", self.result["feature_columns"])
        self.assertEqual(hashlib.sha256(taxi.DATA_PATH.read_bytes()).hexdigest(), self.source_hash)
        # An experiment is unable to read the test split.
        original = pd.read_parquet
        def read(path, *args, **kwargs):
            self.assertNotEqual(Path(path).name, "test.parquet")
            return original(path, *args, **kwargs)
        with patch.object(taxi.pd, "read_parquet", side_effect=read):
            taxi.train_model(self.result)

    def test_frontend_matches_executable_graph(self):
        source = taxi.REPO / "frontend/src/graph/simpleLoopConfig.ts"
        if not source.exists():
            self.skipTest("frontend not mounted; run source parity check on host")
        text = source.read_text()
        nodes = set(re.findall(r"id: '([a-z_]+)'", text))
        edges = set(re.findall(r"source: '([a-z_]+)',\s*target: '([a-z_]+)'", text))
        status = taxi.get_status()
        self.assertEqual(nodes, set(status["nodes"]))
        self.assertEqual(edges, {tuple(e) for e in status["edges"]})


if __name__ == "__main__":
    unittest.main()
