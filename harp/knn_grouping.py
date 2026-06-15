"""
KNN-based grouping utilities for Shapley-valued data valuation.

This module provides functions for:
  - Building balanced groups of data points using cosine-similarity-based
    anchor assignment (replacing KMeans-based grouping).
  - Hierarchical (parent/leaf) group construction for large datasets.
  - Bootstrap stratification of groups for Shapley valuation.
  - Representative leaf group sampling with embedding diversity.
  - Hierarchical leaf-pair sampling (near/mid/far) for pairwise evaluation.

Key public functions:
  - build_groups_from_shapley_ann: Single-level group construction with
    min/max size bounds, optional Lloyd refinement, and pool reclustering.
  - build_hier_groups_from_shapley_ann: Two-level (parent + leaf) hierarchy.
  - sample_representative_leaf_groups: Adaptive size-region + diverse sampling.
  - ann_partition_group_feat_for_bootstrap: Bootstrap stratification.
  - sample_leaf_pairs_hierarchical_buckets: Near/mid/far pair sampling.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


# -----------------------------
# Basic normalization
# -----------------------------
def l2_normalize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    CPU L2 normalize rows. Returns float32.
    """
    X = np.asarray(X, dtype=np.float32)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, eps)


# -----------------------------
# Optional ANN index helpers
# -----------------------------
def build_ann_index_auto(X_unit: np.ndarray, use_gpu: bool = True):
    """
    Build an ANN/NN index over unit-normalized vectors for cosine similarity.

    Returns: (backend, index)
      backend in {"faiss", "sklearn"}

    Notes:
      - FAISS IndexFlatIP is exact NN for cosine on unit vectors.
      - If FAISS GPU is available and use_gpu=True, index is moved to GPU 0.
      - If FAISS not available, falls back to sklearn brute cosine.
    """
    X_unit = np.ascontiguousarray(np.asarray(X_unit, dtype=np.float32))

    # Try FAISS first (exact IP)
    try:
        import faiss
        d = int(X_unit.shape[1])
        index = faiss.IndexFlatIP(d)
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(X_unit)
        return "faiss", index
    except Exception:
        pass

    # Fallback: sklearn brute cosine
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X_unit)
    return "sklearn", nn


def ann_search(backend: str, index, Q_unit: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query ANN index. Returns (idx, sims) with shapes [Q, k].
    sims are cosine similarities in [-1, 1] for both backends.
    """
    Q_unit = np.ascontiguousarray(np.asarray(Q_unit, dtype=np.float32))
    k = int(max(1, k))

    if backend == "faiss":
        sims, idx = index.search(Q_unit, k)  # (D, I)
        return idx.astype(np.int32), sims.astype(np.float32)

    # sklearn returns cosine distance -> similarity = 1 - dist
    dists, idx = index.kneighbors(Q_unit, n_neighbors=k, return_distance=True)
    sims = (1.0 - dists).astype(np.float32)
    return idx.astype(np.int32), sims


# -----------------------------
# Anchor selection + assignment
# -----------------------------
def pick_anchors_farthestish(
    X_unit: np.ndarray,
    n_anchors: int,
    rng: np.random.Generator,
    *,
    cand_pool: int = 2048,
) -> np.ndarray:
    """
    Farthest-ish anchor selection using coverage on a random candidate pool.

    Goal (Shapley): coverage / diversity, not KMeans objective.
    Complexity: O(n_anchors * cand_pool * d)

    Returns: anchors (unique indices into X_unit), sorted.
    """
    X_unit = np.asarray(X_unit, dtype=np.float32)
    n = int(X_unit.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=int)

    n_anchors = int(max(1, min(n_anchors, n)))
    if n_anchors == n:
        return np.arange(n, dtype=int)

    anchors: List[int] = [int(rng.integers(0, n))]
    while len(anchors) < n_anchors:
        cand = rng.choice(n, size=min(int(cand_pool), n), replace=False)
        A = np.asarray(anchors, dtype=int)

        # coverage = max cosine similarity to current anchors
        cover = (X_unit[cand] @ X_unit[A].T).max(axis=1)

        # pick least-covered candidate
        pick = int(cand[int(np.argmin(cover))])
        anchors.append(pick)

        # de-dup in case of rare repeats
        if len(anchors) != len(set(anchors)):
            anchors = list(dict.fromkeys(anchors))

    return np.array(sorted(set(anchors)), dtype=int)


def assign_to_anchor_by_cosine(
    X_unit: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """
    Assign each point to its nearest anchor by cosine similarity.

    Returns:
      anchor_slot: int32 array of shape (n,) with values in [0..len(anchors)-1]
      (i.e., index into `anchors`, not global point id)
    """
    X_unit = np.asarray(X_unit, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=int)
    if anchors.size == 0:
        return np.zeros((X_unit.shape[0],), dtype=np.int32)

    A = X_unit[anchors]              # [A, d]
    sims = X_unit @ A.T              # [n, A]
    return np.argmax(sims, axis=1).astype(np.int32)


# -----------------------------
# Main: build balanced-ish groups
# -----------------------------
def build_groups_from_shapley_ann(
    X_feat: np.ndarray,
    *,
    n_group_clusters: int,
    max_group_size: int,
    rng: np.random.Generator,
    ann_use_gpu: bool = True,   # reserved
    min_group_size: int = 1,
    merge_small_buckets: bool = True,
    refine_steps: int = 1,      # 0 disables; 1 = one Lloyd step in cosine space
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Anchor-bucket -> (optional 1-step spherical Lloyd refinement) ->
    (optional pool tiny buckets) -> bounded split -> (optional recluster pool once).

    Guarantees (best-effort):
      - If feasible, produced group sizes are in [min_group_size, max_group_size].
      - Avoids "last chunk tail" by balanced splitting within each bucket.
      - When merge_small_buckets=True and min_group_size>1:
          * we do NOT union a tiny bucket into another bucket;
          * instead we pool all tiny-bucket points and recluster that pool once.

    Returns:
      groups: list of arrays of point indices (GLOBAL indices)
      group_of_i: (n,) group id for each point
      group_feat: (G,d) normalized mean embedding for each group
    """
    X = np.asarray(X_feat, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X_feat must be 2D, got shape {X.shape}")

    n, d = X.shape
    if n == 0:
        return [], np.zeros((0,), dtype=int), np.zeros((0, d), dtype=np.float32)

    max_group_size = int(max(1, max_group_size))
    min_group_size = int(max(1, min_group_size))
    if min_group_size > max_group_size:
        raise ValueError(f"min_group_size ({min_group_size}) > max_group_size ({max_group_size})")

    refine_steps = int(max(0, refine_steps))

    # cosine geometry
    X_unit = l2_normalize_np(X)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _auto_n_clusters_bounded(n_points: int, mn: int, mx: int) -> int:
        """
        Choose anchor count A so bucket sizes can plausibly end up within [mn, mx]:
          ceil(n/mx) <= A <= floor(n/mn)
        If infeasible, return ceil(n/mx).
        """
        n_points = int(n_points)
        mn = int(max(1, mn))
        mx = int(max(1, mx))
        if n_points <= 1:
            return 1
        lo = int(np.ceil(n_points / float(mx)))
        hi = int(max(1, n_points // mn))
        if lo > hi:
            return int(max(1, min(lo, n_points)))
        return int(max(1, min(hi, lo)))

    def _assign_to_centroids_by_cosine(Xu: np.ndarray, C: np.ndarray, batch: int = 65536) -> np.ndarray:
        """
        Xu: (m,d) unit vectors
        C:  (A,d) unit centroid vectors
        Returns slot: (m,) argmax cosine, in batches to limit memory.
        """
        m = int(Xu.shape[0])
        out = np.empty((m,), dtype=np.int32)
        for s in range(0, m, int(batch)):
            e = min(m, s + int(batch))
            sim = Xu[s:e] @ C.T  # (e-s, A)
            out[s:e] = np.argmax(sim, axis=1).astype(np.int32)
        return out

    def _split_bounded(mem: np.ndarray) -> List[np.ndarray]:
        """
        Split mem into chunks with sizes in [min_group_size, max_group_size] if feasible.
        Balanced sizes (diff at most 1).
        If infeasible, returns [mem].
        """
        mem = np.asarray(mem, dtype=int)
        m = int(mem.size)
        if m == 0:
            return []
        if min_group_size <= m <= max_group_size:
            return [mem.copy()]
        if m < min_group_size:
            return [mem.copy()]

        # feasible number of chunks t:
        t_min = int(np.ceil(m / float(max_group_size)))
        t_max = int(np.floor(m / float(min_group_size)))
        if t_min > t_max:
            return [mem.copy()]

        t = t_min  # fewest chunks => larger chunks
        base = m // t
        rem = m % t
        out: List[np.ndarray] = []
        start = 0
        for k in range(t):
            sz = base + (1 if k < rem else 0)
            out.append(mem[start:start + sz].copy())
            start += sz
        return out

    def _group_indices_from_unit(
        Xsub_unit: np.ndarray,
        *,
        A: int,
        allow_pool_recluster: bool,
    ) -> List[np.ndarray]:
        """
        Group a normalized embedding matrix Xsub_unit (m,d).
        Returns groups as LOCAL indices into [0..m-1].
        If allow_pool_recluster=True and merge_small_buckets and min_group_size>1:
          - pool tiny buckets (<min) and recluster that pool once (no recursion loop).
        """
        m = int(Xsub_unit.shape[0])
        if m == 0:
            return []
        if m == 1:
            return [np.array([0], dtype=int)]

        A = int(max(1, min(int(A), m)))
        if A == 1:
            return [np.arange(m, dtype=int)]

        # 1) anchors + initial assignment
        anchors = pick_anchors_farthestish(Xsub_unit, A, rng)
        bucket_slot = assign_to_anchor_by_cosine(Xsub_unit, anchors)

        buckets: List[List[int]] = [[] for _ in range(A)]
        for i in range(m):
            buckets[int(bucket_slot[i])].append(int(i))

        # 2) optional Lloyd refinement (spherical k-means step)
        if refine_steps > 0:
            C = Xsub_unit[anchors].copy()  # (A,d)

            for _ in range(refine_steps):
                for b in range(A):
                    mem = buckets[b]
                    if len(mem) == 0:
                        continue
                    mu = Xsub_unit[np.asarray(mem, dtype=int)].mean(axis=0)
                    mu = mu / (np.linalg.norm(mu) + 1e-12)
                    C[b] = mu.astype(np.float32)

                bucket_slot = _assign_to_centroids_by_cosine(Xsub_unit, C)

                buckets = [[] for _ in range(A)]
                for i in range(m):
                    buckets[int(bucket_slot[i])].append(int(i))

        # 3) pool tiny buckets (do NOT merge-by-union)
        pool: List[int] = []
        kept: List[List[int]] = []
        if allow_pool_recluster and merge_small_buckets and min_group_size > 1:
            for b in buckets:
                if len(b) == 0:
                    continue
                if len(b) < min_group_size:
                    pool.extend(b)
                else:
                    kept.append(b)
        else:
            kept = [b for b in buckets if len(b) > 0]

        # 4) split kept buckets into bounded groups
        out_groups: List[np.ndarray] = []
        for b in kept:
            mem = np.asarray(b, dtype=int)
            rng.shuffle(mem)
            out_groups.extend(_split_bounded(mem))

        # 5) recluster pooled points once (no recursive pooling loop)
        if allow_pool_recluster and merge_small_buckets and min_group_size > 1 and len(pool) > 0:
            pool_idx = np.asarray(pool, dtype=int)
            rng.shuffle(pool_idx)

            if int(pool_idx.size) < int(min_group_size):
                out_groups.append(pool_idx.copy())
            else:
                A_pool = _auto_n_clusters_bounded(int(pool_idx.size), int(min_group_size), int(max_group_size))

                pool_groups_local = _group_indices_from_unit(
                    Xsub_unit[pool_idx],
                    A=int(A_pool),
                    allow_pool_recluster=False,
                )

                good: List[np.ndarray] = []
                tiny: List[np.ndarray] = []
                for g in pool_groups_local:
                    g = np.asarray(g, dtype=int)
                    if g.size == 0:
                        continue
                    if int(g.size) < int(min_group_size):
                        tiny.append(g)
                    else:
                        good.append(g)

                if len(tiny) > 0:
                    union = np.concatenate(tiny).astype(int, copy=False)
                    rng.shuffle(union)
                    if int(union.size) < int(min_group_size):
                        good.append(union.copy())
                    else:
                        good.extend(_split_bounded(union))

                for g in good:
                    out_groups.append(pool_idx[g].copy())

        return out_groups

    # -----------------------------
    # Top-level grouping
    # -----------------------------
    A = int(max(1, min(int(n_group_clusters), n)))

    # Feasibility cap: don't pick more anchors than min-size can support
    if min_group_size > 1:
        A = min(A, max(1, n // min_group_size))

    groups_local = _group_indices_from_unit(
        X_unit, A=A, allow_pool_recluster=True
    )

    groups: List[np.ndarray] = [np.asarray(g, dtype=int) for g in groups_local if np.asarray(g).size > 0]

    # group_of_i
    group_of_i = -np.ones(n, dtype=np.int32)
    for gid, g in enumerate(groups):
        group_of_i[g] = int(gid)

    # leftover safeguard (should be rare)
    leftover = np.where(group_of_i < 0)[0]
    for i in leftover:
        gid = len(groups)
        groups.append(np.array([int(i)], dtype=int))
        group_of_i[int(i)] = int(gid)

    # group_feat: normalized mean embedding per group
    G = len(groups)
    group_feat = np.zeros((G, d), dtype=np.float32)
    for gid, g in enumerate(groups):
        if g.size == 0:
            continue
        mu = X_unit[g].mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        group_feat[gid] = mu.astype(np.float32)

    return groups, group_of_i.astype(int), group_feat


# -----------------------------
# Bootstrap stratification on group_feat (replaces KMeans fit_predict)
# -----------------------------
def ann_partition_group_feat_for_bootstrap(
    group_feat: np.ndarray,
    k_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Shapley-suited bootstrap stratification:
      - Pick k_boot anchor groups (farthest-ish coverage)
      - Assign every group to nearest anchor (cosine)
      - Output labels in [0..k_boot-1]

    This replaces:
        boot_labels = KMeans(n_clusters=k_boot).fit_predict(group_feat)

    Returns:
      labels: (G,) int array
    """
    GF = l2_normalize_np(np.asarray(group_feat, dtype=np.float32))
    G = int(GF.shape[0])
    if G == 0:
        return np.zeros((0,), dtype=int)
    if G == 1:
        return np.zeros((1,), dtype=int)

    k_boot = int(max(1, min(k_boot, G)))
    if k_boot == 1:
        return np.zeros((G,), dtype=int)

    anchors = pick_anchors_farthestish(GF, k_boot, rng, cand_pool=min(1024, G))
    labels = assign_to_anchor_by_cosine(GF, anchors)  # in [0..k_boot-1]
    return labels.astype(int)


# -----------------------------
# Optional: GPU matmul assignment
# -----------------------------
def assign_to_anchor_by_cosine_torch(
    X_unit: np.ndarray,
    anchors: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """
    Optional speed path: do X @ A^T on GPU with torch.

    Requirements:
      - torch installed
      - a CUDA device available (or set device="cpu")

    Returns: anchor_slot (n,) int32
    """
    import torch
    X_unit = np.asarray(X_unit, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=int)
    if anchors.size == 0:
        return np.zeros((X_unit.shape[0],), dtype=np.int32)

    Xt = torch.from_numpy(X_unit).to(device)
    At = torch.from_numpy(X_unit[anchors]).to(device)
    slot = torch.argmax(Xt @ At.T, dim=1).to("cpu").numpy().astype(np.int32)
    return slot


# ============================================================
# Build PARENT groups (large groups) with anchors
# ============================================================
def build_parents_from_shapley_ann(
    X_feat: np.ndarray,
    *,
    n_parent_clusters: Optional[int] = None,
    parent_max_size: int = 16384,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Build large parent groups by embedding coverage.

    Returns:
      parents: List[np.ndarray] each contains LOCAL indices into X_feat (0..n-1)
      parent_of_i: (n,) parent id per point
      parent_feat: (P, d) normalized mean embedding per parent
    """
    X = np.asarray(X_feat, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X_feat must be 2D, got {X.shape}")
    n, d = int(X.shape[0]), int(X.shape[1])
    if n == 0:
        return [], np.zeros((0,), dtype=int), np.zeros((0, d), dtype=np.float32)

    X_unit = l2_normalize_np(X)

    if n_parent_clusters is None or int(n_parent_clusters) <= 0:
        n_parent_clusters = int(np.ceil(n / max(1, int(parent_max_size))))
        n_parent_clusters = max(2, min(int(n_parent_clusters), n)) if n > 1 else 1
    P = int(n_parent_clusters)

    anchors = pick_anchors_farthestish(X_unit, P, rng)
    parent_slot = assign_to_anchor_by_cosine(X_unit, anchors)  # 0..P-1

    parents: List[np.ndarray] = []
    bins: List[List[int]] = [[] for _ in range(P)]
    for i in range(n):
        bins[int(parent_slot[i])].append(int(i))

    parent_of_i = -np.ones(n, dtype=np.int32)
    for pid in range(P):
        mem = np.asarray(bins[pid], dtype=int)
        if mem.size == 0:
            continue
        parents.append(mem)
        parent_of_i[mem] = len(parents) - 1

    # Handle any unassigned (should not happen)
    leftover = np.where(parent_of_i < 0)[0]
    for i in leftover:
        parents.append(np.array([int(i)], dtype=int))
        parent_of_i[int(i)] = len(parents) - 1

    # parent features
    P2 = len(parents)
    parent_feat = np.zeros((P2, d), dtype=np.float32)
    for pid, mem in enumerate(parents):
        mu = X_unit[mem].mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        parent_feat[pid] = mu.astype(np.float32)

    return parents, parent_of_i.astype(int), parent_feat


# ============================================================
# Build LEAF groups within each parent
# ============================================================
def build_leaf_groups_within_parents(
    X_feat: np.ndarray,
    parents: List[np.ndarray],
    *,
    leaf_max_size: int = 1024,
    leaf_anchor_clusters: Optional[int] = None,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Split each parent into leaf groups.

    Returns (LOCAL indexing w.r.t X_feat):
      leaf_groups: List[np.ndarray] point indices
      leaf_of_i: (n,) leaf gid per point
      leaf_feat: (G, d) normalized mean embedding per leaf
      parent_of_leaf: (G,) parent id for each leaf gid
    """
    X = np.asarray(X_feat, dtype=np.float32)
    n, d = int(X.shape[0]), int(X.shape[1])
    X_unit = l2_normalize_np(X)
    leaf_max_size = int(max(1, leaf_max_size))

    leaf_groups: List[np.ndarray] = []
    leaf_of_i = -np.ones(n, dtype=np.int32)
    parent_of_leaf: List[int] = []

    for pid, mem in enumerate(parents):
        mem = np.asarray(mem, dtype=int)
        if mem.size == 0:
            continue

        # Decide how many anchor buckets inside this parent
        if leaf_anchor_clusters is None or int(leaf_anchor_clusters) <= 0:
            A = int(np.ceil(mem.size / float(leaf_max_size)))
            A = int(max(1, min(A, mem.size)))
        else:
            A = int(max(1, min(int(leaf_anchor_clusters), mem.size)))

        # If only one bucket, just shuffle-chunk
        if A <= 1:
            mm = mem.copy()
            rng.shuffle(mm)
            for s in range(0, mm.size, leaf_max_size):
                g = mm[s:s + leaf_max_size]
                gid = len(leaf_groups)
                leaf_groups.append(g.astype(int))
                leaf_of_i[g] = gid
                parent_of_leaf.append(int(pid))
            continue

        # Otherwise: do a small anchor assignment within parent for locality
        Xp = X_unit[mem]
        anchors_local = pick_anchors_farthestish(Xp, A, rng)
        slot_local = assign_to_anchor_by_cosine(Xp, anchors_local)  # 0..A-1

        # buckets in LOCAL parent coords -> convert back to global ids using mem
        bins: List[List[int]] = [[] for _ in range(A)]
        for j in range(mem.size):
            bins[int(slot_local[j])].append(int(j))

        for b in range(A):
            idxs_local = np.asarray(bins[b], dtype=int)
            if idxs_local.size == 0:
                continue
            mm = mem[idxs_local].copy()
            rng.shuffle(mm)
            for s in range(0, mm.size, leaf_max_size):
                g = mm[s:s + leaf_max_size]
                gid = len(leaf_groups)
                leaf_groups.append(g.astype(int))
                leaf_of_i[g] = gid
                parent_of_leaf.append(int(pid))

    # leftover -> singleton leafs
    leftover = np.where(leaf_of_i < 0)[0]
    for i in leftover:
        gid = len(leaf_groups)
        leaf_groups.append(np.array([int(i)], dtype=int))
        leaf_of_i[int(i)] = gid
        parent_of_leaf.append(-1)

    # fix -1 parents (should be rare)
    parent_of_leaf_arr = np.asarray(parent_of_leaf, dtype=int)
    if np.any(parent_of_leaf_arr < 0):
        curP = int(np.max(parent_of_leaf_arr[parent_of_leaf_arr >= 0])) + 1 if np.any(parent_of_leaf_arr >= 0) else 0
        for gid in range(len(parent_of_leaf_arr)):
            if parent_of_leaf_arr[gid] < 0:
                parent_of_leaf_arr[gid] = curP
                curP += 1

    # leaf features
    G = len(leaf_groups)
    leaf_feat = np.zeros((G, d), dtype=np.float32)
    for gid, g in enumerate(leaf_groups):
        mu = X_unit[g].mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        leaf_feat[gid] = mu.astype(np.float32)

    return leaf_groups, leaf_of_i.astype(int), leaf_feat, parent_of_leaf_arr


# ============================================================
# Wrapper to build hierarchy in ONE call
# ============================================================
def build_hier_groups_from_shapley_ann(
    X_feat: np.ndarray,
    *,
    parent_max_size: int,
    leaf_max_size: int,
    n_parent_clusters: Optional[int],
    leaf_anchor_clusters: Optional[int],
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Returns:
      leaf_groups: List[np.ndarray]
      leaf_of_i: (n,) leaf id per point
      leaf_feat: (G, d)
      parent_of_leaf: (G,)
      parent_groups: List[np.ndarray]
      parent_of_i: (n,)
      parent_feat: (P, d)
    """
    parent_groups, parent_of_i, parent_feat = build_parents_from_shapley_ann(
        X_feat,
        n_parent_clusters=n_parent_clusters,
        parent_max_size=parent_max_size,
        rng=rng,
    )
    leaf_groups, leaf_of_i, leaf_feat, parent_of_leaf = build_leaf_groups_within_parents(
        X_feat,
        parent_groups,
        leaf_max_size=leaf_max_size,
        leaf_anchor_clusters=leaf_anchor_clusters,
        rng=rng,
    )
    return leaf_groups, leaf_of_i, leaf_feat, parent_of_leaf, parent_groups, parent_of_i, parent_feat


# ============================================================
# Build hierarchy on a subset, return GLOBAL ids
# ============================================================
def _build_groups_on_subset_embed(
    X_full: np.ndarray,
    subset_ids: np.ndarray,
    *,
    n_clusters: Optional[int],     # kept for backward compat; maps to n_parent_clusters
    max_sz: int,                   # leaf_max_size
    rng: np.random.Generator,
    parent_max_size: int = 16384,
    leaf_anchor_clusters: Optional[int] = None,
    n_full: Optional[int] = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Build PARENT-FIRST hierarchy on subset, return LEAF groups in GLOBAL ids.

    Returns:
      leaf_groups_global: List[np.ndarray] (GLOBAL point ids)
      leaf_of_i_global: (n_full,) leaf gid per point
      leaf_feat: (G, d)
      parent_of_leaf: (G,) parent id per leaf gid
      parent_groups_global: List[np.ndarray] (GLOBAL point ids)
    """
    subset_ids = np.asarray(subset_ids, dtype=int)
    X_sub = X_full[subset_ids]

    if n_full is None:
        n_full = int(X_full.shape[0])

    n_parent_clusters = None if (n_clusters is None or int(n_clusters) <= 0) else int(n_clusters)

    leaf_groups_local, leaf_of_i_local, leaf_feat, parent_of_leaf, parent_groups_local, parent_of_i_local, parent_feat = build_hier_groups_from_shapley_ann(
        X_sub,
        parent_max_size=int(parent_max_size),
        leaf_max_size=int(max_sz),
        n_parent_clusters=n_parent_clusters,
        leaf_anchor_clusters=leaf_anchor_clusters,
        rng=rng,
    )

    # lift leaf groups to GLOBAL ids
    leaf_groups_global = [subset_ids[np.asarray(g, dtype=int)] for g in leaf_groups_local]
    parent_groups_global = [subset_ids[np.asarray(g, dtype=int)] for g in parent_groups_local]

    # build GLOBAL leaf_of_i mapping
    leaf_of_i_global = np.full(int(n_full), -1, dtype=int)
    for gid, mem in enumerate(leaf_groups_global):
        leaf_of_i_global[np.asarray(mem, dtype=int)] = int(gid)

    return leaf_groups_global, leaf_of_i_global, leaf_feat, np.asarray(parent_of_leaf, dtype=int), parent_groups_global


# ============================================================
# Diverse sampling utilities
# ============================================================
def diverse_sample_kpp(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Robust kmeans++-style diverse sampling.

    Returns:
      chosen: int array of length <= k, indices into X (unique)

    Notes:
      - Samples proportional to current min squared distance to chosen set (kmeans++).
      - Uses a boolean mask for remaining indices (fast, no np.isin).
      - Degenerate / near-identical points fall back to uniform among remaining.
    """
    X = np.asarray(X)
    n = int(X.shape[0])
    k = int(k)
    if n <= 0 or k <= 0:
        return np.zeros((0,), dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)

    # Ensure numeric 2D
    if X.ndim == 1:
        X2 = X.reshape(-1, 1).astype(np.float64, copy=False)
    else:
        X2 = np.asarray(X, dtype=np.float64, order="C")

    chosen = np.empty(k, dtype=int)
    remaining = np.ones(n, dtype=bool)

    # first pick uniformly
    first = int(rng.integers(0, n))
    chosen[0] = first
    remaining[first] = False

    # min squared distance to chosen set
    diff = X2 - X2[first]
    min_d2 = np.einsum("ij,ij->i", diff, diff)

    for t in range(1, k):
        rem_idx = np.flatnonzero(remaining)
        if rem_idx.size == 0:
            return chosen[:t].copy()

        total = float(min_d2[rem_idx].sum())
        if (not np.isfinite(total)) or total <= eps:
            nxt = int(rng.choice(rem_idx))
        else:
            probs = (min_d2[rem_idx] / total).astype(np.float64, copy=False)
            s = float(probs.sum())
            if (not np.isfinite(s)) or s <= eps:
                nxt = int(rng.choice(rem_idx))
            else:
                probs = probs / s
                nxt = int(rng.choice(rem_idx, p=probs))

        chosen[t] = nxt
        remaining[nxt] = False

        diff = X2 - X2[nxt]
        d2_new = np.einsum("ij,ij->i", diff, diff)
        min_d2 = np.minimum(min_d2, d2_new)

    return chosen


def build_parent_to_leaf(parent_of_group: np.ndarray) -> List[np.ndarray]:
    """
    parent_of_group: (G,) parent id for each leaf group gid.
    Returns:
      parent_to_leaf[p] = np.array of leaf gids in parent p
    """
    parent_of_group = np.asarray(parent_of_group, dtype=int)
    if parent_of_group.size == 0:
        return []

    P = int(parent_of_group.max()) + 1
    parent_to_leaf: List[List[int]] = [[] for _ in range(P)]
    for gid, pid in enumerate(parent_of_group.tolist()):
        parent_to_leaf[int(pid)].append(int(gid))

    return [np.asarray(x, dtype=int) for x in parent_to_leaf]


def _choose_size_regions_1d_kmeans(
    svals: np.ndarray,
    *,
    max_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Data-driven size-region assignment via 1D KMeans with BIC-like criterion.

    Returns:
      labels: (n,) int in [0, k-1], where k chosen adaptively in [1..max_bins]
    """
    svals = np.asarray(svals, dtype=np.float64)
    n = int(svals.size)
    if n <= 1:
        return np.zeros((n,), dtype=int)

    uniq = int(np.unique(svals).size)
    Kmax = int(min(max_bins, n, uniq))
    if Kmax <= 1:
        return np.zeros((n,), dtype=int)

    from sklearn.cluster import KMeans

    X = svals.reshape(-1, 1)

    best_labels = None
    best_score = np.inf

    for k in range(1, Kmax + 1):
        km = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).fit(X)
        labels = km.labels_.astype(int)

        centers = km.cluster_centers_[labels, 0]
        rss = float(np.sum((svals - centers) ** 2))
        rss = max(rss, 1e-12)

        score = n * np.log(rss / n) + k * np.log(max(n, 2))
        if score < best_score:
            best_score = score
            best_labels = labels

    labels = best_labels
    k = int(labels.max()) + 1

    # Order region ids by centroid (small -> large)
    cent = np.array([svals[labels == i].mean() for i in range(k)], dtype=np.float64)
    order = np.argsort(cent)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k, dtype=int)
    return remap[labels]


def sample_representative_leaf_groups(
    groups: List[np.ndarray],
    group_feat: np.ndarray,
    parent_of_group: np.ndarray,
    *,
    rep_per_parent: int = 3,
    rng: np.random.Generator,
    use_diverse: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Adaptive size-regions (<= reps) + embedding-diverse selection inside each region.

    Per parent:
      1) compute leaf sizes (RAW)
      2) choose number of size regions adaptively (1..min(rep_per_parent, unique sizes))
      3) allocate r reps across regions by *count* distribution (group-level)
      4) within each region, pick embedding-diverse (KPP) among candidate leaves
    """
    group_feat = np.asarray(group_feat, dtype=np.float32)
    parent_of_group = np.asarray(parent_of_group, dtype=int)

    parent_to_leaf = build_parent_to_leaf(parent_of_group)
    P = len(parent_to_leaf)
    G = len(groups)

    rep_per_parent = int(max(1, rep_per_parent))
    leaf_sizes = np.array([len(g) for g in groups], dtype=np.float64)

    reps_per_parent: List[np.ndarray] = []

    for p in range(P):
        leafs = parent_to_leaf[p]
        if leafs.size == 0:
            reps_per_parent.append(np.zeros((0,), dtype=int))
            continue

        r = int(min(rep_per_parent, leafs.size))
        if r == leafs.size:
            reps_per_parent.append(leafs.astype(int))
            continue

        svals = leaf_sizes[leafs]

        # (1) adaptive size regions
        labels = _choose_size_regions_1d_kmeans(svals, max_bins=r, rng=rng)
        nb = int(labels.max()) + 1

        bins: List[np.ndarray] = []
        for b in range(nb):
            idx = np.where(labels == b)[0]
            bins.append(leafs[idx] if idx.size else np.zeros((0,), dtype=int))

        # (2) allocate reps by COUNT distribution across regions
        weights = np.array([b.size for b in bins], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones(nb, dtype=np.float64)

        frac = weights / weights.sum()
        raw_alloc = frac * float(r)
        alloc = np.floor(raw_alloc).astype(int)

        # ensure each non-empty region gets >=1 if possible
        for b in range(nb):
            if bins[b].size > 0 and alloc[b] == 0 and alloc.sum() < r:
                alloc[b] = 1

        # distribute remaining by largest fractional parts (respect capacity)
        while alloc.sum() < r:
            frac_part = raw_alloc - alloc
            cap = np.array([max(0, bins[b].size - alloc[b]) for b in range(nb)], dtype=int)
            if cap.sum() == 0:
                break
            cand = np.where(cap > 0)[0]
            b_star = int(cand[np.argmax(frac_part[cand])])
            alloc[b_star] += 1

        # if over-allocated (rare), trim from smallest fractional part
        while alloc.sum() > r:
            frac_part = raw_alloc - alloc
            cand = np.where(alloc > 0)[0]
            b_star = int(cand[np.argmin(frac_part[cand])])
            alloc[b_star] -= 1

        # (3) select within each region by embedding diversity
        picked: List[int] = []
        picked_set = set()

        for b in range(nb):
            cand = bins[b]
            k_b = int(min(alloc[b], cand.size))
            if k_b <= 0:
                continue

            if use_diverse and k_b > 1 and cand.size > k_b:
                rel = diverse_sample_kpp(group_feat[cand], k_b, rng)
                sel = cand[rel]
            else:
                sel = rng.choice(cand, size=k_b, replace=False)

            for g in sel.tolist():
                if int(g) not in picked_set:
                    picked_set.add(int(g))
                    picked.append(int(g))

        if len(picked) > r:
            picked = picked[:r]

        reps_per_parent.append(np.asarray(picked, dtype=int))

    rep_leaf_gids = np.array(sorted({int(g) for arr in reps_per_parent for g in arr.tolist()}), dtype=int)
    rep_leaf_gids = rep_leaf_gids[(rep_leaf_gids >= 0) & (rep_leaf_gids < G)]
    return reps_per_parent, rep_leaf_gids


# ============================================================
# Hierarchical leaf-pair sampling (near / mid / far)
# ============================================================
def _sample_two_distinct(arr: np.ndarray, rng: np.random.Generator) -> Optional[Tuple[int, int]]:
    arr = np.asarray(arr, dtype=int)
    if arr.size < 2:
        return None
    a, b = rng.choice(arr, size=2, replace=False)
    return int(a), int(b)


def _sample_one(arr: np.ndarray, rng: np.random.Generator) -> Optional[int]:
    arr = np.asarray(arr, dtype=int)
    if arr.size < 1:
        return None
    return int(rng.choice(arr))


def sample_leaf_pairs_hierarchical_buckets(
    *,
    parent_of_leaf: np.ndarray,          # (G,)
    parent_feat: np.ndarray,             # (P, d) parent centroid embeddings
    leaf_feat: np.ndarray,               # (G, d) leaf centroid embeddings (optional for far tie-break)
    reps_per_parent: Optional[List[np.ndarray]] = None,  # from bootstrap info (optional)
    n_pairs_near: int,
    n_pairs_mid: int,
    n_pairs_far: int,
    rng: np.random.Generator,
    # parent similarity control
    knn_k: int = 64,                     # build parent KNN lists
    mid_rank_lo: int = 8,                # mid parents = KNN ranks [mid_rank_lo .. mid_rank_hi]
    mid_rank_hi: int = 32,
    far_cand_pool: int = 128,            # when choosing far parent q, sample pool from complement and take min sim
    # sampling preferences
    prefer_use_reps: bool = True,        # if True and reps_per_parent provided, draw leaves from reps first
    symmetric_unique: bool = True,       # treat (a,b) and (b,a) as same pair
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Returns dict with keys: "near", "mid", "far".
    Each list contains pairs of LEAF GIDs (gi, gj).

    Definitions:
      near: parent(gi) == parent(gj)
      mid:  parent(gi) != parent(gj) and parent(gj) is in KNN rank bucket [mid_rank_lo..mid_rank_hi]
      far:  parent(gi) != parent(gj) and parent(gj) is chosen from complement of KNN;
            optionally pick most far by min cosine similarity from a pooled candidate set.
    """
    parent_of_leaf = np.asarray(parent_of_leaf, dtype=int)
    PF = np.asarray(parent_feat, dtype=np.float32)
    LF = np.asarray(leaf_feat, dtype=np.float32)
    G = int(parent_of_leaf.size)
    P = int(PF.shape[0])

    out = {"near": [], "mid": [], "far": []}
    if G < 2 or P < 1:
        return out

    parent_to_leaf = build_parent_to_leaf(parent_of_leaf)

    def _parent_leaf_candidates(pid: int) -> np.ndarray:
        base = parent_to_leaf[pid] if (0 <= pid < len(parent_to_leaf)) else np.zeros((0,), dtype=int)
        if not prefer_use_reps or reps_per_parent is None:
            return base
        reps = reps_per_parent[pid] if (0 <= pid < len(reps_per_parent)) else np.zeros((0,), dtype=int)
        reps = np.asarray(reps, dtype=int)
        return reps if reps.size > 0 else base

    # Parent ANN (very fast)
    PF_unit = l2_normalize_np(PF)
    backend, index = build_ann_index_auto(PF_unit, use_gpu=True)

    K = int(max(2, min(knn_k + 1, P)))
    idx_knn, sims_knn = ann_search(backend, index, PF_unit, K)

    def _mid_parents(pid: int) -> np.ndarray:
        nn = idx_knn[pid].astype(int, copy=False).tolist()
        nn = [q for q in nn if q != pid]
        lo = int(max(0, mid_rank_lo))
        hi = int(min(len(nn), mid_rank_hi))
        if hi <= lo:
            return np.zeros((0,), dtype=int)
        return np.asarray(nn[lo:hi], dtype=int)

    def _choose_far_parent(pid: int) -> Optional[int]:
        near = set(int(x) for x in idx_knn[pid].astype(int, copy=False).tolist())
        near.add(int(pid))
        if len(near) >= P:
            return None

        pool_sz = int(min(max(4, far_cand_pool), P - len(near)))
        pool = []
        while len(pool) < pool_sz:
            q = int(rng.integers(0, P))
            if q not in near:
                pool.append(q)
        pool = np.asarray(pool, dtype=int)

        sims = PF_unit[pool] @ PF_unit[pid]
        q = int(pool[int(np.argmin(sims))])
        return q

    used = set()
    def _push(bucket: str, a: int, b: int):
        if a == b:
            return
        if symmetric_unique:
            key = (a, b) if a < b else (b, a)
        else:
            key = (a, b)
        if key in used:
            return
        used.add(key)
        out[bucket].append((int(a), int(b)))

    # 1) NEAR: within same parent
    target = int(n_pairs_near)
    attempts = 0
    cap = max(50 * target, 2000)
    while len(out["near"]) < target and attempts < cap:
        attempts += 1
        pid = int(rng.integers(0, P))
        cand = _parent_leaf_candidates(pid)
        pair = _sample_two_distinct(cand, rng)
        if pair is None:
            continue
        _push("near", pair[0], pair[1])

    # 2) MID: across parents in mid-rank KNN band
    target = int(n_pairs_mid)
    attempts = 0
    cap = max(80 * target, 4000)
    while len(out["mid"]) < target and attempts < cap:
        attempts += 1
        p = int(rng.integers(0, P))
        mids = _mid_parents(p)
        if mids.size == 0:
            continue
        q = int(rng.choice(mids))

        a = _sample_one(_parent_leaf_candidates(p), rng)
        b = _sample_one(_parent_leaf_candidates(q), rng)
        if a is None or b is None:
            continue
        _push("mid", a, b)

    # 3) FAR: across parents in complement of KNN
    target = int(n_pairs_far)
    attempts = 0
    cap = max(120 * target, 8000)
    while len(out["far"]) < target and attempts < cap:
        attempts += 1
        p = int(rng.integers(0, P))
        q = _choose_far_parent(p)
        if q is None:
            continue

        a = _sample_one(_parent_leaf_candidates(p), rng)
        b = _sample_one(_parent_leaf_candidates(q), rng)
        if a is None or b is None:
            continue
        _push("far", a, b)

    return out
