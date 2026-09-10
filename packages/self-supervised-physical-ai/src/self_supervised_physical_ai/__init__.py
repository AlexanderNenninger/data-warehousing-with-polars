"""Unsupervised representation learning for industrial sensor data (physical AI).

A LeJEPA-style (Balestriero & LeCun, arXiv:2511.08544) encoder trained with a
context->target prediction loss plus a SIGReg regularizer, instead of the
EMA/stop-gradient/teacher-student heuristics of classic JEPA.
"""

__version__ = "0.1.0"

from self_supervised_physical_ai.config import HashableBaseModel, LeJEPAConfig
from self_supervised_physical_ai.data import (
    FactoryNetWindowDataset,
    FactoryNetWindowSpec,
    SyntheticSEFCDataset,
)
from self_supervised_physical_ai.encoders import (
    LeJEPAEncoder,
    TransformerStack,
    patch_times,
    patchify,
    time_positional_encoding,
)
from self_supervised_physical_ai.factorynet import (
    FACTORYNET_FILES,
    METADATA_COLUMNS,
    KnownFilesSource,
    SchemaGroup,
    available_metadata_columns,
    build_incremental_pipeline,
    factorynet_source_url,
    group_files_by_schema,
    namespaced_episode_id_expr,
    select_channel_columns,
    source_file_basename,
    split_column_expr,
    split_files_train_test,
    stream_factorynet_to_bucket,
)
from self_supervised_physical_ai.lejepa import LeJEPAModel, LeJEPAOutput, Predictor
from self_supervised_physical_ai.losses import prediction_loss, sigreg_loss
from self_supervised_physical_ai.segmentation import (
    cluster_embeddings,
    embed_dataset,
    label_clusters,
    project_tsne,
)
from self_supervised_physical_ai.views import sample_shared_mask

__all__ = [
    "FACTORYNET_FILES",
    "METADATA_COLUMNS",
    "FactoryNetWindowDataset",
    "FactoryNetWindowSpec",
    "HashableBaseModel",
    "KnownFilesSource",
    "LeJEPAConfig",
    "LeJEPAEncoder",
    "LeJEPAModel",
    "LeJEPAOutput",
    "Predictor",
    "SchemaGroup",
    "SyntheticSEFCDataset",
    "TransformerStack",
    "available_metadata_columns",
    "build_incremental_pipeline",
    "cluster_embeddings",
    "embed_dataset",
    "factorynet_source_url",
    "group_files_by_schema",
    "label_clusters",
    "namespaced_episode_id_expr",
    "patch_times",
    "patchify",
    "prediction_loss",
    "project_tsne",
    "sample_shared_mask",
    "select_channel_columns",
    "sigreg_loss",
    "source_file_basename",
    "split_column_expr",
    "split_files_train_test",
    "stream_factorynet_to_bucket",
    "time_positional_encoding",
]
