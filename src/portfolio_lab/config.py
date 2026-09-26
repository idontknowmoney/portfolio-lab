import math
import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from portfolio_lab.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CONFIG_PATH = CONFIG_DIR / "portfolio.toml"
EXAMPLE_CONFIG_PATH = CONFIG_DIR / "portfolio_example.toml"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Asset(_Model):
    name: str
    isin: str
    ticker: str
    contribution_weight: float = Field(ge=0, le=1)
    initial_value: float = Field(ge=0)
    cost_basis: float = Field(ge=0)
    cost_basis: float = Field(ge=0, default=0)


class PortfolioConfig(_Model):
    name: str
    currency: str
    assets: list[Asset] = Field(min_length=1)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> Self:
        total = sum(a.contribution_weight for a in self.assets)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"asset contribution weights sum to {total}, not 1")
        return self


class SimulationConfig(_Model):
    monthly_contribution: float = Field(ge=0)
    horizon_years: int = Field(gt=0)
    n_paths: int = Field(gt=0)
    block_size: int = Field(gt=0)
    seed: int
    rebalance: str


class Config(_Model):
    portfolio: PortfolioConfig
    simulation: SimulationConfig


def _resolve_path(path: Path, fallback: Path) -> Path:
    """
    Pick the config file to load: the given path, else the fallback with a warning.

    Raises:
    FileNotFoundError: If neither file exists.
    """
    if path.is_file():
        return path
    if fallback.is_file():
        logger.warning("Config file %s not found, falling back to %s", path, fallback)
        return fallback
    raise FileNotFoundError(f"Neither {path} nor the example config {fallback} exist")


def load_config(path: Path = CONFIG_PATH, fallback: Path = EXAMPLE_CONFIG_PATH) -> Config:
    """
    Load the configuration from a TOML file, falling back to the example file
    (with a warning) if it is not present.

    Parameters:
    path (Path): Path to the configuration file.
    fallback (Path): Path to the example configuration used when `path` is missing.

    Returns:
    Config: The parsed configuration.

    Raises:
    FileNotFoundError: If neither file exists.
    ValueError: If the file fails validation (missing keys, wrong types, weights not summing to 1).
    """
    resolved = _resolve_path(path, fallback)
    with resolved.open("rb") as f:
        raw = tomllib.load(f)

    try:
        config = Config.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid config file {resolved}:\n{e}") from e

    logger.info("Loaded config from %s", resolved)
    return config
