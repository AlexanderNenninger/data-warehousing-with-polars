"""Config models for self_supervised_physical_ai."""

from pydantic import BaseModel, ConfigDict


class HashableBaseModel(BaseModel):
    """A frozen, hashable pydantic BaseModel.

    Hashes by serializing the model to canonical JSON, so hashability extends
    to nested models/collections without every field type needing its own
    __hash__.
    """

    model_config = ConfigDict(frozen=True)

    def __hash__(self) -> int:
        return hash(self.model_dump_json())


class LeJEPAConfig(HashableBaseModel):
    """Hyperparameters for LeJEPA training.

    See Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised
    Learning Without the Heuristics" (arXiv:2511.08544).
    """

    # --- data / windowing ---
    n_channels: int = 8  # S-E-F-C channels after preprocessing; adjust to real schema
    window_len: int = 512  # timesteps per training window
    patch_len: int = 16  # timesteps per patch/token (window_len must be divisible by patch_len)

    # --- encoder ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1

    # --- predictor (smaller than the encoder, as in I-JEPA/LeJEPA) ---
    predictor_d_model: int = 64
    predictor_n_heads: int = 4
    predictor_n_layers: int = 2

    # --- masking (context/target split over patches) ---
    context_ratio: float = 0.6  # fraction of patches visible to the context encoder
    n_target_blocks: int = 4  # number of contiguous target blocks sampled per window

    # --- SIGReg ---
    sigreg_n_directions: int = 64  # random 1D projections per step (sketching)
    sigreg_n_freqs: int = 8  # quadrature points for the characteristic-function test
    sigreg_lambda: float = 1.0  # single LeJEPA trade-off hyperparameter (pred loss vs. SIGReg)

    # --- optimization ---
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 10

    @property
    def n_patches(self) -> int:
        assert self.window_len % self.patch_len == 0, "window_len must be divisible by patch_len"
        return self.window_len // self.patch_len
