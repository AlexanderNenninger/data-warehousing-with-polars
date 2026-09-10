"""Tests for the config models."""

import pytest
from self_supervised_physical_ai import LeJEPAConfig


def test_lejepa_config_defaults():
    cfg = LeJEPAConfig()
    assert cfg.n_patches == cfg.window_len // cfg.patch_len


def test_lejepa_config_hashable_and_frozen():
    a = LeJEPAConfig(d_model=64)
    b = LeJEPAConfig(d_model=64)
    c = LeJEPAConfig(d_model=128)
    assert hash(a) == hash(b)
    assert hash(a) != hash(c)
    assert {a, b, c} == {a, c}
    with pytest.raises(Exception):
        a.d_model = 1
