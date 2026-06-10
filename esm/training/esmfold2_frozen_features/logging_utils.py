from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def _rank_from_env() -> int:
    for key in ("RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return 0


def is_main_process() -> bool:
    return _rank_from_env() == 0


class _NullTqdm:
    def __init__(self, iterable: Iterable[Any] | None = None, **_: Any) -> None:
        self.iterable = iterable

    def __iter__(self) -> Iterator[Any]:
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def __enter__(self) -> "_NullTqdm":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def update(self, _: int = 1) -> None:
        return None

    def set_postfix(self, *_: Any, **__: Any) -> None:
        return None

    def close(self) -> None:
        return None


def maybe_tqdm(
    iterable: Iterable[Any] | None = None, *, enabled: bool, **kwargs: Any
) -> Any:
    if enabled and is_main_process():
        from tqdm.auto import tqdm

        return tqdm(iterable, **kwargs)
    return _NullTqdm(iterable, **kwargs)


def _metric_value(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _format_float(value: Any, fmt: str) -> str:
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def format_tqdm_postfix(metrics: dict[str, Any]) -> dict[str, str]:
    fields = (
        ("loss", ("train/loss", "loss"), ".4g"),
        ("struct", ("train/loss_struct", "loss_struct"), ".4g"),
        ("dist", ("train/loss_dist", "loss_dist"), ".3g"),
        ("conf", ("train/loss_conf", "loss_conf"), ".4g"),
        ("lr", ("opt/lr", "lr"), ".2e"),
        ("grad_norm", ("opt/grad_norm", "grad_norm"), ".3g"),
    )
    result: dict[str, str] = {}
    for label, keys, fmt in fields:
        value = _metric_value(metrics, *keys)
        if value is not None:
            result[label] = _format_float(value, fmt)
    return result


class JsonlLogger:
    def __init__(self, path: str | Path, *, enabled: bool, mode: str = "w") -> None:
        self.path = Path(path)
        self.enabled = enabled and is_main_process()
        self.mode = mode
        self._handle: Any = None

    def __enter__(self) -> "JsonlLogger":
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open(self.mode)
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def write(self, record: dict[str, Any]) -> bool:
        if self._handle is None:
            return False
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()
        return True

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
