"""Trusted resolution and setup of official Ultralytics detection weights."""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Literal, Optional, Tuple


LOGGER = logging.getLogger(__name__)
YOLO26_MINIMUM_VERSION = (8, 4, 0)
OFFICIAL_MODEL_NAMES = {
    "yolo26n.pt", "yolo26s.pt", "yolo11n.pt", "yolo11s.pt",
}


class ModelSetupError(RuntimeError):
    """Expected model selection, download, or validation failure."""


@dataclass(frozen=True)
class HardwareInfo:
    torch_version: str
    cuda_available: bool
    mps_available: bool


@dataclass(frozen=True)
class ResolvedPlayerModel:
    path: Path
    model_name: str
    device: str
    source_type: Literal["official", "custom"]
    ultralytics_version: str


def inspect_hardware(torch_module: Optional[Any] = None) -> HardwareInfo:
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise ModelSetupError(
                "PyTorch is required for player detection. Install the volleyball "
                "dependencies with: pip install -e '.[volleyball]'"
            ) from exc
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    return HardwareInfo(
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        cuda_available=bool(torch_module.cuda.is_available()),
        mps_available=bool(mps_backend and mps_backend.is_available()),
    )


def select_device(requested: str, hardware: HardwareInfo) -> str:
    if requested != "auto":
        return requested
    if hardware.cuda_available:
        return "cuda:0"
    if hardware.mps_available:
        return "mps"
    return "cpu"


def _version_tuple(version: str) -> Tuple[int, int, int]:
    numbers = [int(value) for value in re.findall(r"\d+", version)[:3]]
    return tuple((numbers + [0, 0, 0])[:3])  # type: ignore[return-value]


def choose_auto_model(ultralytics_version: str, device: str) -> str:
    """Choose the newest model family supported by the installed package."""
    generation = "yolo26" if _version_tuple(
        ultralytics_version) >= YOLO26_MINIMUM_VERSION else "yolo11"
    scale = "s" if device.startswith("cuda") else "n"
    return f"{generation}{scale}.pt"


def _load_ultralytics() -> Tuple[Any, str]:
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelSetupError(
            "Ultralytics is required for player detection. Install the volleyball "
            "dependencies with: pip install -e '.[volleyball]'"
        ) from exc
    return YOLO, str(ultralytics.__version__)


def _ensure_official_model(YOLO: Any, model_name: str, models_dir: Path) -> Path:
    if model_name not in OFFICIAL_MODEL_NAMES:
        raise ModelSetupError(
            f"'{model_name}' is not an allowed official detection model. Allowed: "
            + ", ".join(sorted(OFFICIAL_MODEL_NAMES)))
    models_dir = models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / model_name
    if model_path.is_file() and model_path.stat().st_size > 0:
        LOGGER.info("Using cached official player model: %s", model_path)
        return model_path

    LOGGER.info("Official player detection model not found locally.")
    LOGGER.info("Model: %s", model_name)
    LOGGER.info("Source: Ultralytics official model assets")
    LOGGER.info("Downloading to: %s", model_path)
    try:
        YOLO(str(model_path))
    except Exception as exc:
        raise ModelSetupError(
            f"Could not download/load official Ultralytics model '{model_name}': {exc}"
        ) from exc
    if not model_path.is_file() or model_path.stat().st_size <= 0:
        raise ModelSetupError(
            "Ultralytics completed without creating the expected model file at "
            f"'{model_path}'. Upgrade Ultralytics or pass a trusted local .pt path."
        )
    LOGGER.info("Official model ready: %s", model_path)
    return model_path


def resolve_player_model(
    selection: str,
    *,
    models_dir: Path = Path("models/ultralytics"),
    requested_device: str = "auto",
    torch_module: Optional[Any] = None,
    yolo_class: Optional[Any] = None,
    ultralytics_version: Optional[str] = None,
) -> ResolvedPlayerModel:
    """Resolve a custom local path or trusted official model to a local file."""
    hardware = inspect_hardware(torch_module)
    device = select_device(requested_device, hardware)
    if yolo_class is None or ultralytics_version is None:
        installed_yolo, installed_version = _load_ultralytics()
        yolo_class = yolo_class or installed_yolo
        ultralytics_version = ultralytics_version or installed_version

    if selection == "auto":
        model_name = choose_auto_model(ultralytics_version, device)
        path = _ensure_official_model(yolo_class, model_name, models_dir)
        source_type: Literal["official", "custom"] = "official"
    elif selection in OFFICIAL_MODEL_NAMES:
        model_name = selection
        path = _ensure_official_model(yolo_class, model_name, models_dir)
        source_type = "official"
    else:
        path = Path(selection).expanduser().resolve()
        if not path.is_file():
            raise ModelSetupError(
                f"Custom player model was not found at '{path}'. Use 'auto', an "
                "allowed official name, or an existing trusted local model path."
            )
        model_name = path.name
        source_type = "custom"

    LOGGER.info(
        "Player model: %s (%s), device=%s, path=%s",
        model_name, source_type, device, path)
    return ResolvedPlayerModel(
        path=path,
        model_name=model_name,
        device=device,
        source_type=source_type,
        ultralytics_version=ultralytics_version,
    )
