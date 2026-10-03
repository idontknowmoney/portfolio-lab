import logging
from pathlib import Path

import pytest

from portfolio_lab.config import EXAMPLE_CONFIG_PATH, load_config

VALID = """
[portfolio]
name = "Test"
currency = "EUR"

[[portfolio.assets]]
name = "A"
isin = "X"
ticker = "A.DE"
contribution_weight = 1.0
initial_value = 10
cost_basis = 8

[simulation]
monthly_contribution = 100
horizon_years = 5
n_paths = 10
seed = 1
block_size = 3
rebalance = "none"
"""


def test_loads_primary_file(tmp_path: Path):
    path = tmp_path / "portfolio.toml"
    path.write_text(VALID)
    config = load_config(path, tmp_path / "missing.toml")
    assert config.portfolio.name == "Test"
    assert config.portfolio.assets[0].ticker == "A.DE"
    assert config.simulation.n_paths == 10


def test_falls_back_to_example_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    logging.getLogger("portfolio_lab").propagate = True
    with caplog.at_level(logging.WARNING):
        config = load_config(tmp_path / "missing.toml", EXAMPLE_CONFIG_PATH)
    assert config.portfolio.assets
    assert "falling back" in caplog.text


def test_raises_when_no_file_exists(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "a.toml", tmp_path / "b.toml")


def test_rejects_bad_weights(tmp_path: Path):
    path = tmp_path / "portfolio.toml"
    path.write_text(VALID.replace("contribution_weight = 1.0", "contribution_weight = 0.5"))
    with pytest.raises(ValueError, match="sum to 0.5"):
        load_config(path, tmp_path / "missing.toml")


def test_rejects_missing_keys(tmp_path: Path):
    path = tmp_path / "portfolio.toml"
    path.write_text(VALID.replace("seed = 1\n", ""))
    with pytest.raises(ValueError, match="seed"):
        load_config(path, tmp_path / "missing.toml")


def test_rejects_wrong_types(tmp_path: Path):
    path = tmp_path / "portfolio.toml"
    path.write_text(VALID.replace("seed = 1", 'seed = "abc"'))
    with pytest.raises(ValueError, match="seed"):
        load_config(path, tmp_path / "missing.toml")


def test_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "portfolio.toml"
    path.write_text(VALID.replace("seed = 1", "seed = 1\nn_path = 3"))
    with pytest.raises(ValueError, match="n_path"):
        load_config(path, tmp_path / "missing.toml")


PROXY = """
proxy = [
  { ticker = "SPY", currency = "USD", weight = 0.7 },
  { ticker = "EFA", currency = "USD", weight = 0.3 },
]
"""


def test_proxy_defaults(tmp_path: Path):
    path = tmp_path / "portfolio.toml"
    path.write_text(VALID)
    config = load_config(path, tmp_path / "missing.toml")
    assert config.portfolio.assets[0].proxy == []
    assert config.simulation.use_proxies is True


def test_loads_proxy_blend(tmp_path: Path):
    path = tmp_path / "portfolio.toml"
    path.write_text(VALID.replace("cost_basis = 8", "cost_basis = 8" + PROXY))
    proxy = load_config(path, tmp_path / "missing.toml").portfolio.assets[0].proxy
    assert [c.ticker for c in proxy] == ["SPY", "EFA"]


def test_rejects_proxy_weights_not_summing_to_one(tmp_path: Path):
    path = tmp_path / "portfolio.toml"
    path.write_text(VALID.replace("cost_basis = 8", "cost_basis = 8" + PROXY.replace("0.3", "0.2")))
    with pytest.raises(ValueError, match="proxy weights sum to"):
        load_config(path, tmp_path / "missing.toml")


def test_example_config_has_proxies():
    config = load_config(EXAMPLE_CONFIG_PATH, EXAMPLE_CONFIG_PATH)
    assert all(a.proxy for a in config.portfolio.assets)
