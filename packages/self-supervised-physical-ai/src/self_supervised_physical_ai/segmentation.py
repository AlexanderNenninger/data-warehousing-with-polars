"""Turning a trained encoder's embeddings into labeled segments.

Three steps, each a thin function so they compose in a notebook: embed every
window with the (already-trained) LeJEPA encoder, cluster the embeddings
(KMeans — a first pass at "segmenting" the data into operating regimes, not a
claim that regimes are convex/globular), then attach a human-readable label to
each cluster by majority vote over a metadata column you already have (e.g.
`ctx_anomaly_label`, `machine_type`) rather than inventing one.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from self_supervised_physical_ai.lejepa import LeJEPAModel


def embed_dataset(
    model: LeJEPAModel,
    dataset: Dataset,
    metadata_columns: Sequence[str] = (),
    batch_size: int = 128,
    device: str = "cpu",
) -> tuple[np.ndarray, pl.DataFrame]:
    """Mean-pools each window's patch embeddings (full, unmasked encoder pass)
    into one vector per window, carrying along the requested metadata columns
    for later cluster labeling.

    Returns:
        (embeddings, metadata) — `embeddings` is `(n_windows, d_model)`;
        `metadata` has one row per window with `metadata_columns` as columns.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings: list[np.ndarray] = []
    metadata: dict[str, list] = {col: [] for col in metadata_columns}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            sample_rate_hz = batch["sample_rate_hz"].to(device)
            patch_emb = model.encoder(x, sample_rate_hz)  # (b, n_patches, d_model)
            embeddings.append(patch_emb.mean(dim=1).cpu().numpy())
            for col in metadata_columns:
                values = batch[col]
                metadata[col].extend(values.tolist() if torch.is_tensor(values) else list(values))

    embeddings_arr = np.concatenate(embeddings, axis=0)
    metadata_df = (
        pl.DataFrame(metadata)
        if metadata_columns
        else pl.DataFrame({"_row": np.arange(len(embeddings_arr))})
    )
    return embeddings_arr, metadata_df


def cluster_embeddings(
    embeddings: np.ndarray, n_clusters: int = 8, random_state: int = 0
) -> np.ndarray:
    """KMeans over the embeddings — a first pass at segmenting windows into
    operating regimes. Returns one cluster id per row of `embeddings`."""
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit_predict(
        embeddings
    )


def project_tsne(
    embeddings: np.ndarray, random_state: int = 0, perplexity: float = 30.0
) -> np.ndarray:
    """2D t-SNE projection of `embeddings`, for plotting the discovered clusters."""
    from sklearn.manifold import TSNE

    # t-SNE requires perplexity < n_samples; clamp so small demo runs don't error.
    perplexity = min(perplexity, max(5.0, (len(embeddings) - 1) / 3))
    return TSNE(
        n_components=2, random_state=random_state, perplexity=perplexity, init="pca"
    ).fit_transform(embeddings)


def label_clusters(
    cluster_ids: np.ndarray, metadata: pl.DataFrame, label_column: str
) -> dict[int, str]:
    """Majority-vote label per cluster from a metadata column (e.g. `ctx_anomaly_label`).

    The vote share is included in the label (e.g. `"Anomaly (83%)"`) so it's
    never presented as more confident than the majority actually was.
    """
    df = metadata.with_columns(pl.Series("cluster", cluster_ids))
    labels: dict[int, str] = {}
    for cluster_id in sorted(df["cluster"].unique().to_list()):
        counts = (
            df.filter(pl.col("cluster") == cluster_id)
            .group_by(label_column)
            .len()
            .sort("len", descending=True)
        )
        value, count = counts.row(0)
        total = counts["len"].sum()
        labels[int(cluster_id)] = f"{value} ({count / total:.0%})"
    return labels
