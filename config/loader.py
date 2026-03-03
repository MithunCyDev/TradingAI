"""
Central configuration loader for HQTS.

Loads config/settings.yaml and merges with .env overrides.
Used by training scripts and data fetchers; does not replace ExecutionConfig.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore


def _env_str(key: str, default: Optional[str]) -> Optional[str]:
    val = os.getenv(key)
    return val if val is not None else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).lower() in ("true", "1", "yes")


@dataclass
class PathsConfig:
    """Path configuration."""

    models_dir: str = "models"
    data_dir: str = "data/clean"
    config_file: str = "config/settings.yaml"


@dataclass
class DataConfig:
    """Data source and range configuration."""

    source: str = "dukascopy"
    symbols: list[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD", "XAGUSD"])
    timeframes: list[str] = field(default_factory=lambda: ["15m", "1h", "4h", "1d"])
    start_year: int = 2004
    end_year: int = 2026


@dataclass
class WalkForwardConfig:
    """Walk-forward training window configuration."""

    train_years: int = 10
    test_years: int = 1
    step_years: int = 1
    mode: str = "rolling"  # rolling | expanding


@dataclass
class TripleBarrierConfig:
    """Triple-barrier labeling parameters."""

    atr_mult_sl: float = 1.0
    rr_ratio: float = 2.0
    horizon_bars: int = 16
    vertical_barrier_bars: int = 16


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""

    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    triple_barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    meta_labeling: bool = True
    regime_aware: bool = True
    hyperopt: bool = True
    hyperopt_trials: int = 50


@dataclass
class AppConfig:
    """Full application configuration for training and data."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file; return empty dict if not found or yaml unavailable."""
    if yaml is None:
        return {}
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Apply .env overrides to config."""
    if _env_str("MODELS_BASE_DIR", None):
        config.paths.models_dir = _env_str("MODELS_BASE_DIR", config.paths.models_dir) or config.paths.models_dir
    if _env_str("DATA_DIR", None):
        config.paths.data_dir = _env_str("DATA_DIR", config.paths.data_dir) or config.paths.data_dir
    return config


def _dict_to_config(data: dict[str, Any]) -> AppConfig:
    """Convert raw dict to AppConfig dataclass."""
    paths_data = data.get("paths", {})
    paths = PathsConfig(
        models_dir=str(paths_data.get("models_dir", "models")),
        data_dir=str(paths_data.get("data_dir", "data/clean")),
        config_file=str(paths_data.get("config_file", "config/settings.yaml")),
    )

    data_cfg = data.get("data", {})
    symbols = data_cfg.get("symbols", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD", "XAGUSD"])
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",")]
    data_config = DataConfig(
        source=str(data_cfg.get("source", "dukascopy")),
        symbols=symbols,
        timeframes=list(data_cfg.get("timeframes", ["15m", "1h", "4h", "1d"])),
        start_year=int(data_cfg.get("start_year", 2004)),
        end_year=int(data_cfg.get("end_year", 2026)),
    )

    train_data = data.get("training", {})
    wf_data = train_data.get("walk_forward", {})
    wf = WalkForwardConfig(
        train_years=int(wf_data.get("train_years", 10)),
        test_years=int(wf_data.get("test_years", 1)),
        step_years=int(wf_data.get("step_years", 1)),
        mode=str(wf_data.get("mode", "rolling")),
    )
    tb_data = train_data.get("triple_barrier", {})
    tb = TripleBarrierConfig(
        atr_mult_sl=float(tb_data.get("atr_mult_sl", 1.0)),
        rr_ratio=float(tb_data.get("rr_ratio", 2.0)),
        horizon_bars=int(tb_data.get("horizon_bars", 16)),
        vertical_barrier_bars=int(tb_data.get("vertical_barrier_bars", 16)),
    )
    training = TrainingConfig(
        walk_forward=wf,
        triple_barrier=tb,
        meta_labeling=bool(train_data.get("meta_labeling", True)),
        regime_aware=bool(train_data.get("regime_aware", True)),
        hyperopt=bool(train_data.get("hyperopt", True)),
        hyperopt_trials=int(train_data.get("hyperopt_trials", 50)),
    )

    return AppConfig(paths=paths, data=data_config, training=training)


def get_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from YAML and merge with .env overrides.

    Args:
        config_path: Path to settings.yaml. If None, uses project root / config/settings.yaml.

    Returns:
        AppConfig dataclass with merged values.
    """
    if load_dotenv:
        load_dotenv()

    project_root = Path(__file__).resolve().parent.parent
    path = config_path or project_root / "config" / "settings.yaml"
    if not path.is_absolute():
        path = project_root / path

    raw = _load_yaml(path)
    config = _dict_to_config(raw)
    config = _apply_env_overrides(config)
    return config
