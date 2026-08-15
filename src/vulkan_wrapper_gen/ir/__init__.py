"""Middle-layer IR: processed, comprehensive registry data.

The IR is the single source of truth between the raw Khronos registry XML
(vk.xml, video.xml, ...) and the C++ emitter.  It normalizes array/count
relationships, parameter direction, receivers, member names, output shapes,
creation records and releasers while keeping every raw attribute and doc
comment, and it reproduces the exact C API signature of each command for
internal dispatch.

:func:`build_ir` produces the IR and the emitter consumes it directly, so the
generator no longer re-derives analysis from the raw XML.  The IR is also
JSON-serializable (:meth:`IrRegistry.to_json` / :meth:`IrRegistry.from_json`)
so it can be inspected, consumed by other tools, or emitted by the CLI via
``--emit-ir``.
"""

from .builder import RegistryError, build_ir
from .model import (
    Alias,
    Availability,
    Basetype,
    Bitmask,
    Command,
    Constant,
    Define,
    Enum,
    EnumValue,
    FuncPointer,
    Handle,
    IrRegistry,
    Length,
    Param,
    RawType,
    Struct,
)

__all__ = [
    "Alias",
    "Availability",
    "Basetype",
    "Bitmask",
    "Command",
    "Constant",
    "Define",
    "Enum",
    "EnumValue",
    "FuncPointer",
    "Handle",
    "IrRegistry",
    "Length",
    "Param",
    "RawType",
    "RegistryError",
    "Struct",
    "build_ir",
]
