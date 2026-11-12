#!/usr/bin/env python3
"""Command-line interface for CuraCastBench evaluation, retrieval, and checks."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def run_check() -> None:
    from curacast.eval import evaluate

    # Proposition 1: query selection reverses ranker order at matched budget B = 3
    task = {
        "queries": ["q1", "q2", "q3"],
        "targets": ["y1", "y2", "y3", "y4"],
        "known": [],
        "events": [["q1", "y1"], ["q1", "y2"], ["q_new", "y1"]],
    }
    ranker_a_full = {"q1": ["y1", "y3", "y4", "y2"], "q2": ["y3"], "q3": ["y3"]}
    ranker_b_full = {"q1": ["y3", "y1", "y2", "y4"], "q2": ["y3"], "q3": ["y3"]}
    ranker_a_sel = {"q1": ["y1", "y3", "y4", "y2"], "q2": [], "q3": []}
    ranker_b_sel = {"q1": ["y3", "y1", "y2", "y4"], "q2": [], "q3": []}

    full_a = evaluate(task, ranker_a_full, [3])[0]
    full_b = evaluate(task, ranker_b_full, [3])[0]
    sel_a = evaluate(task, ranker_a_sel, [3])[0]
    sel_b = evaluate(task, ranker_b_sel, [3])[0]

    assert full_a["hits"] == 1 and full_b["hits"] == 0
    assert sel_a["hits"] == 1 and sel_b["hits"] == 2
    assert abs(full_a["recall"] - full_a["pool_recall"] * full_a["inspection_reach"]) < 1e-12

    summary = {
        "proposition_1_reversal": True,
        "decomposition_identity": True,
        "full_population_hits": {"ranker_a": full_a["hits"], "ranker_b": full_b["hits"]},
        "selected_query_hits": {"ranker_a": sel_a["hits"], "ranker_b": sel_b["hits"]},
    }
    print(json.dumps(summary, indent=2))


def run_demo() -> None:
    from curacast.eval import evaluate, graph_lists

    tasks = json.loads((ROOT / "data" / "demo.json").read_text())
    results = []
    for task in tasks:
        n = len(task["queries"])
        query_set = set(task["queries"])
        target_set = set(task["targets"])
        methods = {}
        for name in ("degree", "coannotation", "adamic_adar"):
            lists = graph_lists(task, method=name, depth=10)
            rows = evaluate(task, lists, [n, 3 * n, 10 * n])
            methods[name] = {r["budget"]: r["hits"] for r in rows}
        results.append(
            {
                "task": task["id"],
                "cutoff": task["cutoff"],
                "destination": task["destination"],
                "queries": n,
                "eligible": len(
                    {(q, y) for q, y in task["events"] if q in query_set and y in target_set}
                ),
                "hits_by_budget": methods,
            }
        )
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Verify protocol invariants and rank-reversal propositions")
    sub.add_parser("demo", help="Evaluate graph rankers on GitHub Advisory release transitions")

    p_eval = sub.add_parser("eval", help="Score ranked or empty candidate lists for a task")
    p_eval.add_argument("--task", type=Path, required=True)
    p_eval.add_argument("--lists", type=Path, required=True)
    p_eval.add_argument("--budgets", type=int, nargs="+", required=True)
    p_eval.add_argument("--seed", type=int, default=20260711)

    p_train = sub.add_parser("train", help="Train reaction-difference retriever on cutoff features")
    p_train.add_argument("--data", type=Path, required=True)
    p_train.add_argument("--out", type=Path, required=True)
    p_train.add_argument("--seed", type=int, default=20261112)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch", type=int, default=1024)
    p_train.add_argument("--device", default="cuda")

    p_infer = sub.add_parser("infer", help="Generate full-catalog top-k predictions from a checkpoint")
    p_infer.add_argument("--data", type=Path, required=True)
    p_infer.add_argument("--checkpoint", type=Path, required=True)
    p_infer.add_argument("--out", type=Path, required=True)
    p_infer.add_argument("--depth", type=int, default=100)
    p_infer.add_argument("--batch", type=int, default=2048)
    p_infer.add_argument("--device", default="cuda")

    p_bench = sub.add_parser("bench", help="Benchmark retrieval operators on query and target vectors")
    for name in ("queries", "targets", "known", "valid"):
        p_bench.add_argument(f"--{name}", type=Path, required=True)
    p_bench.add_argument("--operator", choices=["gpu", "cpu", "csr", "sql", "ann"], default="cpu")
    p_bench.add_argument("--depth", type=int, default=10)
    p_bench.add_argument("--batch", type=int, default=2048)
    p_bench.add_argument("--threads", type=int, default=8)

    args = parser.parse_args()
    if args.command == "check":
        run_check()
    elif args.command == "demo":
        run_demo()
    elif args.command == "eval":
        from curacast.eval import evaluate

        task = json.loads(args.task.read_text())
        lists = json.loads(args.lists.read_text())
        print(json.dumps(evaluate(task, lists, args.budgets, args.seed), indent=2))
    elif args.command == "train":
        from curacast.model import fit

        fit(args.data, args.out, args.seed, args.epochs, args.batch, args.device)
    elif args.command == "infer":
        import numpy as np
        from curacast.model import retrieve

        preds = retrieve(args.data, args.checkpoint, args.depth, args.batch, args.device)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, preds, allow_pickle=False)
        print(json.dumps({"queries": len(preds), "depth": args.depth, "output": str(args.out)}))
    elif args.command == "bench":
        import numpy as np
        from curacast.ops import search

        inputs = [np.load(getattr(args, k), mmap_mode="r") for k in ("queries", "targets", "known", "valid")]
        t0 = time.perf_counter()
        got = search(*inputs, args.operator, args.depth, args.batch, args.threads)
        print(
            json.dumps(
                {
                    "operator": args.operator,
                    "queries": len(got),
                    "depth": args.depth,
                    "seconds": round(time.perf_counter() - t0, 4),
                }
            )
        )


if __name__ == "__main__":
    main()
