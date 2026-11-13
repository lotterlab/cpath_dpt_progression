"""
Ordering Baselines for Pseudotime Comparison.

Two families of methods benchmark against Diffusion Pseudotime (DPT):

Naive baselines (no neighbor graph):
1. Centroid Distance: cosine distance from the earliest-class centroid.
2. PCA Projection: projection onto the first principal component.

DPT ablations (share DPT's kNN graph):
3. Diffusion Component 1: ordering along the leading non-trivial eigenvector.
4. kNN Geodesic: shortest-path distance from the root medoid.

The ablations strip one ingredient of DPT while holding the others fixed, to
isolate whether DPT's performance comes from the graph geometry, the diffusion
smoothing, or the root-anchored multi-component distance. They are NOT naive
baselines and should not be plotted as such.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scanpy as sc
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

from analysis.dpt import find_medoid

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Results from an ordering baseline."""
    method: str = ""
    tau: float = np.nan
    p_value: float = np.nan
    # Fraction of samples the ordering could score. Only knn_geodesic sets this
    # to anything but 1.0, when the kNN graph is disconnected from the root.
    frac_valid: float = 1.0

    @property
    def is_valid(self) -> bool:
        return not np.isnan(self.tau)


# =============================================================================
# Per-patch scores
#
# Every method assigns each patch a scalar that increases along the
# progression. Kendall's tau against stage_int is one summary of that vector;
# cross-method Spearman agreement is another. Both analyses read the SAME
# score functions below, so an agreement result can never drift from the tau
# result it is reported alongside.
# =============================================================================

def _tau_from_score(
    score: Optional[np.ndarray],
    adata: sc.AnnData,
    method: str,
    frac_valid: float = 1.0,
) -> BaselineResult:
    """Summarise a per-patch score vector as Kendall's tau against stage_int."""
    result = BaselineResult(method=method, frac_valid=frac_valid)

    if score is None:
        return result

    valid = np.isfinite(score)
    if valid.sum() < 10:
        return result

    tau, p_val = kendalltau(score[valid], adata.obs["stage_int"].values[valid])
    result.tau = tau
    result.p_value = p_val
    return result


def score_centroid_distance(
    adata: sc.AnnData, root_class: str
) -> Optional[np.ndarray]:
    """Per-patch cosine distance from the mean of the root_class embeddings."""
    mask = adata.obs["class"] == root_class
    if mask.sum() == 0:
        logger.warning(f"Root class '{root_class}' not found in data.")
        return None

    centroid = adata.X[mask].mean(axis=0).reshape(1, -1)
    return cdist(adata.X, centroid, metric="cosine").flatten()


def score_pca_projection(adata: sc.AnnData) -> Optional[np.ndarray]:
    """
    Per-patch projection onto PC1.

    NOTE: the sign is fitted to stage_int -- the same labels the projection is
    later scored against -- so this method's tau cannot go negative. This
    differs from the root-anchored orientation used by DC1 and is retained
    only for continuity with the published baseline numbers.
    """
    X = adata.X.copy()
    X_centered = X - X.mean(axis=0)

    try:
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    projections = X_centered @ Vt[0]

    # Orient: flip sign if negatively correlated with stage order
    tau_raw, _ = kendalltau(projections, adata.obs["stage_int"].values)
    if tau_raw < 0:
        projections = -projections

    return projections


def compute_centroid_distance(adata: sc.AnnData, root_class: str) -> BaselineResult:
    """
    Baseline: cosine distance from the earliest-class centroid.

    Args:
        adata: AnnData with X = embeddings, obs containing 'class' and 'stage_int'.
        root_class: Name of the earliest class (e.g. "Epidermis").

    Returns:
        BaselineResult with tau and p_value.
    """
    score = score_centroid_distance(adata, root_class)
    return _tau_from_score(score, adata, "centroid_distance")


def compute_pca_projection(adata: sc.AnnData) -> BaselineResult:
    """
    Baseline: projection onto the first principal component.

    Args:
        adata: AnnData with X = embeddings, obs containing 'stage_int'.

    Returns:
        BaselineResult with tau and p_value.
    """
    score = score_pca_projection(adata)
    return _tau_from_score(score, adata, "pca_projection")


# =============================================================================
# DPT Ablations
#
# Both methods below build the SAME neighbor graph as compute_dpt() in
# analysis/dpt.py (cosine metric, gauss kernel, use_rep="X", identical
# n_neighbors). Any divergence in these parameters would confound the ablation.
# =============================================================================

def prepare_neighbor_graph(adata: sc.AnnData, n_neighbors: int) -> None:
    """
    Build the kNN graph in-place, matching compute_dpt's construction exactly.

    Idempotent: returns immediately if a graph is already present. This lets a
    caller build the graph once and have several ablations reuse it, while each
    ablation remains independently callable.
    """
    if "neighbors" in adata.uns:
        return

    sc.pp.neighbors(
        adata,
        use_rep="X",
        n_neighbors=n_neighbors,
        metric="cosine",
        method="gauss",
    )


def compute_diffusion_component(
    adata: sc.AnnData, root_class: str, n_neighbors: int
) -> BaselineResult:
    """
    Ablation: ordering along the first non-trivial diffusion component (DC1).

    This is DPT minus the root-anchored distance through the full diffusion
    space -- it keeps the graph and the diffusion smoothing, but reduces the
    ordering to a single coordinate.

    The sign of an eigenvector is arbitrary, so DC1 is oriented so the root
    class sits at the low end. This uses only root_class, exactly the
    information DPT receives via `iroot`, so tau can legitimately be negative
    when the leading component does not track the progression.

    Args:
        adata: AnnData with X = embeddings, obs containing 'class' and 'stage_int'.
        root_class: Name of the earliest class (e.g. "Epidermis").
        n_neighbors: Neighbors for graph construction (must match DPT's).

    Returns:
        BaselineResult with tau and p_value.
    """
    score = score_diffusion_component(adata, root_class, n_neighbors)
    return _tau_from_score(score, adata, "diffusion_component_1")


def score_diffusion_component(
    adata: sc.AnnData, root_class: str, n_neighbors: int
) -> Optional[np.ndarray]:
    """Per-patch DC1 coordinate, sign-oriented so the root class sits low."""
    if (adata.obs["class"] == root_class).sum() == 0:
        logger.warning(f"Root class '{root_class}' not found in data.")
        return None

    try:
        prepare_neighbor_graph(adata, n_neighbors)
        if "X_diffmap" not in adata.obsm:
            sc.tl.diffmap(adata)
    except Exception as e:
        logger.warning(f"Diffusion map failed: {e}")
        return None

    if "X_diffmap" not in adata.obsm:
        return None

    diffmap = adata.obsm["X_diffmap"]
    if diffmap.shape[1] < 2:
        return None

    # Column 0 is the trivial steady-state eigenvector (eval == 1.0); the first
    # informative component is column 1. Consistent with _compute_spectral_gap
    # in analysis/dpt.py, which treats evals[0] as the steady state.
    dc1 = diffmap[:, 1]

    valid = np.isfinite(dc1)
    if valid.sum() < 10:
        return dc1

    # Root-anchor the sign: the root class should sit at the low end.
    root_mask = (adata.obs["class"] == root_class).values & valid
    if root_mask.sum() > 0 and dc1[root_mask].mean() > dc1[valid].mean():
        dc1 = -dc1

    return dc1


def compute_knn_geodesic(
    adata: sc.AnnData, root_class: str, n_neighbors: int
) -> BaselineResult:
    """
    Ablation: ordering by shortest-path distance along the kNN graph.

    This is DPT minus the diffusion smoothing -- it keeps the graph and the root
    anchoring, but walks raw edge weights instead of diffusing over them.

    The root is the medoid of root_class, the same cell DPT assigns to `iroot`.
    A disconnected graph is not treated as an error: tau is computed over the
    nodes reachable from the root, and `frac_valid` records that coverage so a
    tau derived from a fragmented graph is visible rather than silent.

    Args:
        adata: AnnData with X = embeddings, obs containing 'class' and 'stage_int'.
        root_class: Name of the earliest class (e.g. "Epidermis").
        n_neighbors: Neighbors for graph construction (must match DPT's).

    Returns:
        BaselineResult with tau, p_value, and frac_valid.
    """
    score = score_knn_geodesic(adata, root_class, n_neighbors)
    frac_valid = 1.0 if score is None else float(np.isfinite(score).mean())
    return _tau_from_score(score, adata, "knn_geodesic", frac_valid)


def score_knn_geodesic(
    adata: sc.AnnData, root_class: str, n_neighbors: int
) -> Optional[np.ndarray]:
    """
    Per-patch shortest-path distance from the root medoid along the kNN graph.

    Nodes unreachable from the root come back as inf rather than being dropped,
    so the caller can decide how to treat a disconnected graph.
    """
    try:
        prepare_neighbor_graph(adata, n_neighbors)
    except Exception as e:
        logger.warning(f"Neighbor graph failed: {e}")
        return None

    root_idx = find_medoid(adata, root_class)
    if root_idx is None:
        logger.warning(f"Root class '{root_class}' medoid not found.")
        return None

    if "distances" not in adata.obsp:
        return None

    try:
        geodesic = dijkstra(
            adata.obsp["distances"], directed=False, indices=root_idx
        )
    except Exception as e:
        logger.warning(f"Dijkstra failed: {e}")
        return None

    frac_valid = float(np.isfinite(geodesic).mean())
    if frac_valid < 1.0:
        logger.warning(
            f"kNN graph disconnected: only {frac_valid:.1%} of nodes "
            f"reachable from the root. Tau computed on that subset."
        )

    return geodesic


def score_dpt(
    adata: sc.AnnData, root_class: str, n_neighbors: int, n_dcs: int
) -> Optional[np.ndarray]:
    """
    Per-patch DPT pseudotime.

    Mirrors compute_dpt() in analysis/dpt.py: same graph, same diffusion map,
    same medoid root, same n_dcs. Exposed here so cross-method agreement can be
    measured against DPT on the identical cohort without re-running the full
    metric suite.
    """
    try:
        prepare_neighbor_graph(adata, n_neighbors)
        if "X_diffmap" not in adata.obsm:
            sc.tl.diffmap(adata)
    except Exception as e:
        logger.warning(f"Diffusion map failed: {e}")
        return None

    root_idx = find_medoid(adata, root_class)
    if root_idx is None:
        logger.warning(f"Root class '{root_class}' medoid not found.")
        return None

    adata.uns["iroot"] = root_idx

    try:
        sc.tl.dpt(adata, n_dcs=n_dcs)
    except Exception as e:
        logger.warning(f"DPT failed: {e}")
        return None

    if "dpt_pseudotime" not in adata.obs:
        return None

    return np.asarray(adata.obs["dpt_pseudotime"].values, dtype=float)
