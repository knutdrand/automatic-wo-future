"""Test fixtures for disease model."""

import pytest


@pytest.fixture
def model_config():
    """Create a test model configuration."""
    from main import DiseaseModelConfig

    return DiseaseModelConfig(lags=6, lags_past_covariates=6, n_samples=10)
