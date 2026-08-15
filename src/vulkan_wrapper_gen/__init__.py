"""Vulkan C++ wrapper generator."""

from .config import GeneratorConfig, load_config
from .ir import IrRegistry, RegistryError, build_ir
from .template import TemplateError, render_template

__all__ = [
    "GeneratorConfig",
    "IrRegistry",
    "RegistryError",
    "TemplateError",
    "build_ir",
    "load_config",
    "render_template",
]

__version__ = "0.1.0"

