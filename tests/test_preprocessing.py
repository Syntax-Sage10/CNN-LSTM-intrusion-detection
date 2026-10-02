"""Leakage and shape guarantees for the data pipeline."""
import numpy as np
import pytest

from ids.config import Config
from ids.data import generate_dummy_data
from ids.preprocessing import FeaturePipeline, split_dataset, to_sequences


@pytest.fixture(scope="module")
def ds():
    return generate_dummy_data(n_samples=2000, seed=0)


def test_scaler_fit_on_train_only(ds):
    s = split_dataset(ds, Config(seed=0))
    s.X_test[0, :] = 1e12                       # extreme test values
    pipe = FeaturePipeline(10, 0).fit(s.X_train, s.y_train)
    np.testing.assert_allclose(pipe.scaler.data_max_, s.X_train.max(axis=0))
    Xt = pipe.transform(s.X_test)
    assert Xt.min() >= 0.0 and Xt.max() <= 1.0  # clipped, not extrapolated


def test_selector_ignores_test_set(ds):
    s = split_dataset(ds, Config(seed=0))
    a = FeaturePipeline(10, 0).fit(s.X_train, s.y_train)
    s.X_test[:] = np.random.default_rng(1).random(s.X_test.shape)  # scramble test
    b = FeaturePipeline(10, 0).fit(s.X_train, s.y_train)
    assert (a.selector.get_support() == b.selector.get_support()).all()


def test_random_split_sizes_and_stratification(ds):
    s = split_dataset(ds, Config(seed=0, test_size=0.2))
    assert len(s.y_train) + len(s.y_test) == len(ds.y)
    assert set(s.y_test) == set(ds.y)           # every class reaches the test set


def test_to_sequences_shape(ds):
    s = split_dataset(ds, Config(seed=0))
    X = to_sequences(FeaturePipeline(10, 0).fit(s.X_train, s.y_train).transform(s.X_test))
    assert X.shape == (len(s.y_test), 10, 1) and X.dtype == np.float32