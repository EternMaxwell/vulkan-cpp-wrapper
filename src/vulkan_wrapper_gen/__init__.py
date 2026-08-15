"""Vulkan C++ wrapper generator."""

from .config import GeneratorConfig, load_config
from .ir import IrRegistry, build_ir
from .registry import Registry, RegistryError, parse_registries
from .template import TemplateError, render_template

__all__ = [
    "GeneratorConfig",
    "IrRegistry",
    "Registry",
    "RegistryError",
    "TemplateError",
    "build_ir",
    "load_config",
    "parse_registries",
    "render_template",
]

__version__ = "0.1.0"

