"""Tests for embedding, clustering, projecting, and labeling — using the
synthetic dataset's `regime` field as a stand-in for real FactoryNet metadata
(no network access)."""

import numpy as np
from self_supervised_physical_ai import (
    LeJEPAConfig,
    LeJEPAModel,
    SyntheticSEFCDataset,
    cluster_embeddings,
    embed_dataset,
    label_clusters,
    project_tsne,
)


def _tiny_config() -> LeJEPAConfig:
    return LeJEPAConfig(
        n_channels=2,
        window_len=32,
        patch_len=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_mult=2,
        dropout=0.0,
        predictor_d_model=8,
        predictor_n_heads=4,
        predictor_n_layers=1,
        context_ratio=0.5,
        n_target_blocks=1,
    )


def test_embed_dataset_shapes_and_metadata():
    cfg = _tiny_config()
    model = LeJEPAModel(cfg)
    dataset = SyntheticSEFCDataset(
        n_windows=30, n_channels=cfg.n_channels, window_len=cfg.window_len
    )

    embeddings, metadata = embed_dataset(model, dataset, metadata_columns=("regime",), batch_size=8)

    assert embeddings.shape == (30, cfg.d_model)
    assert np.isfinite(embeddings).all()
    assert metadata.height == 30
    assert set(metadata["regime"].unique().to_list()) <= {0, 1, 2}


def test_cluster_project_and_label_round_trip():
    cfg = _tiny_config()
    model = LeJEPAModel(cfg)
    dataset = SyntheticSEFCDataset(
        n_windows=30, n_channels=cfg.n_channels, window_len=cfg.window_len
    )
    embeddings, metadata = embed_dataset(model, dataset, metadata_columns=("regime",), batch_size=8)

    cluster_ids = cluster_embeddings(embeddings, n_clusters=3, random_state=0)
    assert cluster_ids.shape == (30,)
    assert set(cluster_ids.tolist()) <= {0, 1, 2}

    coords_2d = project_tsne(embeddings, random_state=0)
    assert coords_2d.shape == (30, 2)
    assert np.isfinite(coords_2d).all()

    labels = label_clusters(cluster_ids, metadata, label_column="regime")
    assert set(labels.keys()) == set(cluster_ids.tolist())
    for label in labels.values():
        assert label.endswith("%)")
