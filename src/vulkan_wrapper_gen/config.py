from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


class ConfigError(ValueError):
    pass


@dataclass(slots=True)
class ReceiverOverride:
    add: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    rename: str | None = None


@dataclass(slots=True)
class GeneratorConfig:
    version: int = 1
    namespace: str = "vk"
    module: str = "vulkan.wrapper"
    api: str = "vulkan"
    minimum_core: str = "1.3"
    include_extensions: tuple[str, ...] = ("*",)
    exclude_extensions: tuple[str, ...] = ()
    exclude_commands: tuple[str, ...] = ()
    exclude_types: tuple[str, ...] = ()
    receivers: dict[str, ReceiverOverride] = field(default_factory=dict)
    type_names: dict[str, str] = field(default_factory=dict)
    command_names: dict[str, str] = field(default_factory=dict)
    vma_include: str = "vk_mem_alloc.h"
    vma_functions: tuple[str, ...] = (
        "vmaCreateAllocator", "vmaDestroyAllocator", "vmaAllocateMemory",
        "vmaFreeMemory", "vmaMapMemory", "vmaUnmapMemory",
        "vmaFlushAllocation", "vmaInvalidateAllocation", "vmaGetAllocationInfo",
        "vmaCreateBuffer", "vmaDestroyBuffer", "vmaCreateImage", "vmaDestroyImage",
    )


def _strings(value: object, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{key} must be an array of strings")
    return tuple(value)


def load_config(path: Path | None = None) -> GeneratorConfig:
    if path is None:
        return GeneratorConfig()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot load configuration {path}: {exc}") from exc
    version = data.get("version", 1)
    if version != 1:
        raise ConfigError(f"unsupported configuration version {version!r}; expected 1")
    project = data.get("generator", {})
    filters = data.get("filters", {})
    naming = data.get("naming", {})
    vma = data.get("vma", {})
    if not all(isinstance(x, dict) for x in (project, filters, naming, vma)):
        raise ConfigError("generator, filters, naming, and vma must be TOML tables")
    overrides: dict[str, ReceiverOverride] = {}
    receiver_data = data.get("receivers", {})
    if not isinstance(receiver_data, dict):
        raise ConfigError("receivers must be a TOML table")
    for command, raw in receiver_data.items():
        if not isinstance(raw, dict):
            raise ConfigError(f"receivers.{command} must be a table")
        overrides[command] = ReceiverOverride(
            _strings(raw.get("add"), f"receivers.{command}.add"),
            _strings(raw.get("remove"), f"receivers.{command}.remove"),
            raw.get("rename"),
        )
    type_names = naming.get("types", {})
    command_names = naming.get("commands", {})
    if not isinstance(type_names, dict) or not isinstance(command_names, dict):
        raise ConfigError("naming.types and naming.commands must be tables")
    return GeneratorConfig(
        namespace=str(project.get("namespace", "vk")),
        module=str(project.get("module", "vulkan.wrapper")),
        api=str(project.get("api", "vulkan")),
        minimum_core=str(project.get("minimum_core", "1.3")),
        include_extensions=_strings(filters.get("include_extensions", ["*"]), "filters.include_extensions"),
        exclude_extensions=_strings(filters.get("exclude_extensions"), "filters.exclude_extensions"),
        exclude_commands=_strings(filters.get("exclude_commands"), "filters.exclude_commands"),
        exclude_types=_strings(filters.get("exclude_types"), "filters.exclude_types"),
        receivers=overrides,
        type_names={str(k): str(v) for k, v in type_names.items()},
        command_names={str(k): str(v) for k, v in command_names.items()},
        vma_include=str(vma.get("include", "vk_mem_alloc.h")),
        vma_functions=_strings(vma.get("functions", list(GeneratorConfig().vma_functions)), "vma.functions"),
    )
