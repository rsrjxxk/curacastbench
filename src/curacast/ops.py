"""Physical pair-scoring operators for full-catalog top-k retrieval."""

import numpy as np
import torch


def search(
    query: np.ndarray,
    target: np.ndarray,
    known: np.ndarray,
    valid: np.ndarray,
    operator: str = "cpu",
    depth: int = 10,
    batch: int = 2048,
    threads: int = 8,
) -> np.ndarray:
    """Execute exact (gpu, cpu, csr, sql) or approximate (ann) masked top-k search."""
    from .model import top_indices

    query = np.asarray(query, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    known = np.asarray(known, dtype=np.int64).reshape(-1, 2)
    known = known[np.argsort(known[:, 0], kind="stable")]
    valid = np.asarray(valid, dtype=bool)

    if query.ndim != 2 or target.ndim != 2 or query.shape[1] != target.shape[1]:
        raise ValueError("Query and target vector dimensions must match")
    if valid.shape != (len(target),) or not (0 < depth < len(target)):
        raise ValueError("Invalid depth or target validity mask")
    if len(known) and (known.min() < 0 or known[:, 0].max() >= len(query) or known[:, 1].max() >= len(target)):
        raise ValueError("Known pair index out of bounds")

    bounds = np.searchsorted(known[:, 0], np.arange(len(query) + 1))
    blocked = [np.unique(known[bounds[i] : bounds[i + 1], 1]) for i in range(len(query))]
    allowed = np.array([int(valid.sum()) - int(valid[ids].sum()) for ids in blocked])
    if np.any(allowed < depth):
        raise ValueError("Fewer valid unknown targets than requested depth")

    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if operator == "gpu" else "cpu"

    if operator in ("gpu", "cpu"):
        bank = torch.from_numpy(target.T.copy()).to(device)
    elif operator == "ann":
        import faiss

        faiss.omp_set_num_threads(threads)
        active = np.flatnonzero(valid)
        dim = target.shape[1]
        nlist = min(256, max(16, len(active) // 40))
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, 32, 8, faiss.METRIC_INNER_PRODUCT)
        index.cp.seed = 20261112
        index.pq.cp.seed = 20261112
        index.train(target[active])
        index.add(target[active])
        index.nprobe = min(32, nlist)
    elif operator == "sql":
        import duckdb
        import pandas as pd

        connection = duckdb.connect()
        connection.execute(f"SET threads={int(threads)}")
        frame = pd.DataFrame({"tid": np.flatnonzero(valid), "vec": target[valid].tolist()})
        connection.register("target_frame", frame)
        connection.execute(
            f"CREATE TABLE target AS SELECT tid::INTEGER tid, vec::FLOAT[{target.shape[1]}] vec FROM target_frame"
        )
    elif operator != "csr":
        raise ValueError(f"Unknown operator: {operator}")

    output = np.empty((len(query), depth), dtype=np.int32)
    for begin in range(0, len(query), batch):
        end = min(begin + batch, len(query))
        block = np.asarray(query[begin:end]).copy()

        if operator in ("gpu", "cpu", "csr"):
            if operator == "csr":
                from scipy import sparse

                score = torch.from_numpy(sparse.csr_matrix(block) @ target.T)
            else:
                score = torch.from_numpy(block).to(device) @ bank
            score[:, torch.as_tensor(~valid, device=score.device)] = -torch.inf
            pairs = known[bounds[begin] : bounds[end]]
            if len(pairs):
                ix = torch.as_tensor(pairs, device=score.device)
                score[ix[:, 0] - begin, ix[:, 1]] = -torch.inf
            ids = top_indices(score, depth).cpu().numpy()
        elif operator == "sql":
            frame = pd.DataFrame(
                {
                    "qid": np.arange(begin, end),
                    "vec": block.tolist(),
                    "blocked": [known[bounds[i] : bounds[i + 1], 1].tolist() for i in range(begin, end)],
                }
            )
            connection.register("query_frame", frame)
            dim = target.shape[1]
            rows = connection.execute(
                f"""
                SELECT qid, arg_max(tid, struct_pack(s := score, t := -tid), {depth}) AS ids
                FROM (
                    SELECT qid, tid, array_inner_product(query_frame.vec::FLOAT[{dim}], target.vec) AS score
                    FROM query_frame CROSS JOIN target
                    WHERE NOT list_contains(blocked, tid)
                ) scored
                GROUP BY qid
                ORDER BY qid
                """
            ).fetchall()
            ids = np.asarray([row[1] for row in rows], dtype=np.int32)
        else:
            _, found = index.search(block, min(256, len(active)))
            selected = []
            for i, row in enumerate(found):
                choices = active[row[row >= 0]]
                choices = choices[~np.isin(choices, known[bounds[begin + i] : bounds[begin + i + 1], 1])]
                if len(choices) < depth:
                    raise ValueError("ANN shortlist exhausted before reaching requested depth")
                selected.append(choices[:depth])
            ids = np.asarray(selected, dtype=np.int32)

        output[begin:end] = ids

    if operator == "sql":
        connection.close()
    return output
