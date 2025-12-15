"""Tests for disease model using chap-python-sdk."""

import pytest

from main import DiseaseModelConfig, on_predict, on_train, runner


class TestDiseaseModel:
    """Tests for the disease model implementation."""

    @pytest.mark.asyncio
    async def test_model_with_sdk_validation(self) -> None:
        """Test model validation using chap-python-sdk."""
        from chap_python_sdk.testing import get_example_data, validate_model_io

        example_data = get_example_data(country="laos", frequency="monthly")
        config = DiseaseModelConfig(lags=6, lags_past_covariates=6, n_samples=10)

        result = await validate_model_io(runner, example_data, config)

        assert result.success, f"Validation failed: {result.errors}"
        assert result.n_predictions > 0, "No predictions generated"
        assert result.n_samples >= 1, "No samples in predictions"

    @pytest.mark.asyncio
    async def test_train_produces_models(self) -> None:
        """Test that training produces models for locations."""
        from chap_python_sdk.testing import get_example_data

        example_data = get_example_data(country="laos", frequency="monthly")
        config = DiseaseModelConfig(lags=6, lags_past_covariates=6, n_samples=10)

        trained_model = await on_train(config, example_data.training_data, None)

        assert "models" in trained_model, "No models in training output"
        assert "training_stats" in trained_model, "No training stats"
        assert len(trained_model["models"]) > 0, "No models trained"

    @pytest.mark.asyncio
    async def test_predictions_have_correct_format(self) -> None:
        """Test that predictions have correct output format."""
        from chap_python_sdk.testing import get_example_data

        example_data = get_example_data(country="laos", frequency="monthly")
        config = DiseaseModelConfig(lags=6, lags_past_covariates=6, n_samples=10)

        trained_model = await on_train(config, example_data.training_data, None)
        predictions = await on_predict(
            config,
            trained_model,
            example_data.historic_data,
            example_data.future_data,
            None,
        )

        pred_df = predictions.to_pandas()

        assert "time_period" in pred_df.columns, "Missing time_period column"
        assert "location" in pred_df.columns, "Missing location column"
        assert "samples" in pred_df.columns, "Missing samples column"

        # Check samples are lists with correct length
        first_samples = pred_df["samples"].iloc[0]
        assert isinstance(first_samples, list), "Samples should be a list"
        assert len(first_samples) == config.n_samples, f"Expected {config.n_samples} samples"
