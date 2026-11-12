"""Budgeted link prediction evaluation over versioned cutoff catalogs."""

from collections import Counter, defaultdict
import hashlib
import math
from typing import Iterable


def stable_query_order(queries: Iterable[str], seed: int) -> list[str]:
    """Deterministic seeded query order for incomplete round-robin waves."""
    unique = set(queries)
    prefix = f"curacast-operational-v1\x00{seed}\x00".encode("ascii")
    return sorted(
        unique,
        key=lambda q: (hashlib.sha256(prefix + q.encode("ascii")).digest(), q),
    )


def evaluate(task: dict, rankings: dict, budgets: Iterable[int], seed: int = 20260711) -> list[dict]:
    """Score ranked candidate lists under round-robin shared-budget inspection."""
    queries = task["queries"]
    targets = task["targets"]
    if len(set(queries)) != len(queries) or len(set(targets)) != len(targets):
        raise ValueError("Duplicate catalog identifiers")

    qs, ts = set(queries), set(targets)
    if set(rankings) != qs:
        raise ValueError("Rankings must cover every cutoff query")

    known = {tuple(p) for p in task.get("known", [])}
    events = {tuple(p) for p in task["events"]}
    gold = {(q, y) for q, y in events if q in qs and y in ts}

    pool = set()
    for q, ys in rankings.items():
        if len(set(ys)) != len(ys):
            raise ValueError(f"Duplicate targets in list for query {q}")
        for y in ys:
            if y not in ts:
                raise ValueError(f"Target {y} absent from cutoff catalog")
            if (q, y) in known:
                raise ValueError(f"Known pair ({q}, {y}) proposed")
            pool.add((q, y))

    caps = sorted(set(map(int, budgets)))
    if any(b < 0 for b in caps):
        raise ValueError("Budgets must be non-negative")

    order = stable_query_order(queries, seed)
    active = [q for q in order if rankings[q]]
    examined = 0
    hits = 0
    recovered = set()
    hit_at = {0: (0, set())}
    depth = 0
    cap = max(caps, default=0)

    while active and examined < cap:
        next_active = []
        for q in active:
            if examined == cap:
                break
            y = rankings[q][depth]
            examined += 1
            if (q, y) in gold:
                hits += 1
                recovered.add((q, y))
            if examined in caps:
                hit_at[examined] = (hits, set(recovered))
            if len(rankings[q]) > depth + 1:
                next_active.append(q)
        active = next_active
        depth += 1

    hit_at[examined] = (hits, set(recovered))
    target_counts = Counter(y for _, y in gold)
    covered = len(pool & gold)

    rows = []
    for b in caps:
        actual = min(b, len(pool))
        h, rec = hit_at[actual]
        rows.append(
            {
                "budget": b,
                "actual_inspections": actual,
                "hits": h,
                "eligible": len(gold),
                "recall": h / len(gold) if gold else None,
                "pool_hits": covered,
                "pool_recall": covered / len(gold) if gold else None,
                "inspection_reach": h / covered if covered else 0.0,
                "endpoint_availability": len(gold) / len(events) if events else None,
                "unavailable_endpoints": len(events) - len(gold),
                "omitted_candidates": len(gold) - covered,
                "listed_uninspected": covered - h,
                "unique_targets": len({y for _, y in rec}),
                "target_macro_recall": (
                    sum(1.0 / target_counts[y] for _, y in rec) / len(target_counts)
                    if target_counts
                    else None
                ),
            }
        )
    return rows


def graph_lists(task: dict, method: str = "degree", depth: int = 10) -> dict[str, list[str]]:
    """Build cutoff bipartite graph rankings (degree, coannotation, adamic_adar)."""
    queries = task["queries"]
    targets = task["targets"]
    query_set = set(queries)
    t_idx = {y: i for i, y in enumerate(targets)}

    q_neighbors: dict[str, list[int]] = defaultdict(list)
    t_queries: list[list[str]] = [[] for _ in targets]
    for q, y in sorted({tuple(p) for p in task.get("known", [])}):
        if q in query_set and y in t_idx:
            yi = t_idx[y]
            q_neighbors[q].append(yi)
            t_queries[yi].append(q)

    dt = [len(qs) for qs in t_queries]
    fallback_idx = sorted(range(len(targets)), key=lambda i: (-dt[i], i))

    if method == "degree":
        out = {}
        for q in queries:
            blocked = set(q_neighbors[q])
            out[q] = [targets[i] for i in fallback_idx if i not in blocked][:depth]
        return out

    q_deg = {q: len(ys) for q, ys in q_neighbors.items()}
    sim: list[dict[int, float]] = [defaultdict(float) for _ in targets]
    if method == "coannotation":
        for q, ys in q_neighbors.items():
            for s in ys:
                for y in ys:
                    if s != y:
                        sim[s][y] += 1.0
        for s in range(len(targets)):
            denom_s = math.sqrt(max(dt[s], 1))
            for y in list(sim[s]):
                sim[s][y] /= denom_s * math.sqrt(max(dt[y], 1))
    elif method == "adamic_adar":
        for q, ys in q_neighbors.items():
            w = 1.0 / max(math.log1p(q_deg[q]), 1.0)
            for s in ys:
                for y in ys:
                    sim[s][y] += w
    else:
        raise ValueError(f"Unknown graph ranking method: {method}")

    out = {}
    for q in queries:
        known_y = q_neighbors[q]
        if not known_y:
            out[q] = [targets[i] for i in fallback_idx[:depth]]
            continue
        blocked = set(known_y)
        scores: dict[int, float] = defaultdict(float)
        for s in known_y:
            for y, val in sim[s].items():
                if y not in blocked:
                    scores[y] += val
        ranked = sorted(
            (i for i in fallback_idx if i not in blocked),
            key=lambda i: (-scores.get(i, 0.0), -dt[i], i),
        )
        out[q] = [targets[i] for i in ranked[:depth]]
    return out


def degree_lists(task: dict, depth: int = 10) -> dict[str, list[str]]:
    """Rank cutoff targets by global degree with known-pair exclusion."""
    return graph_lists(task, method="degree", depth=depth)
