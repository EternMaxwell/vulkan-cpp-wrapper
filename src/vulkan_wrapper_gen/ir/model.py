"""Middle-layer IR model for the Vulkan wrapper generator.

The IR is a processed, registry-neutral representation of Khronos registry XML
input (vk.xml, video.xml, ...).  Compared to the raw XML it:

* normalizes ``len``/``altlen``/count parameters into explicit array
  relationships (:attr:`Param.lengths`, :attr:`Param.counts_for`,
  :attr:`Param.is_array`),
* classifies every parameter/member by direction (input/output),
* keys every entity collection by its processed general name (``Buffer``,
  ``createBuffer``) and keeps the C API spelling in ``c_name`` (``VkBuffer``,
  ``vkCreateBuffer``); every reference to another entity also uses general
  names,
* resolves aliases, parents, receivers, member-function names, output shapes,
  creation records and releasers,
* keeps every raw XML attribute (optional, externsync, selection, protect,
  availability, docs/comments, ...) so no input information is lost,
* and can reproduce the exact C API signature of every command
  (:attr:`Command.c_signature`) for internal dispatch.

Every IR object can be serialized to/from JSON (:meth:`IrRegistry.to_json`,
:meth:`IrRegistry.from_json`), which makes the middle layer inspectable and
usable by other tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Name mapping helpers
# ---------------------------------------------------------------------------

# C primitive spellings that never gain a ``Vk`` prefix when reconstructed.
_PRIMITIVE_C_TYPES = frozenset({
    "void", "char", "float", "double", "bool", "int", "short", "long",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "size_t", "intptr_t", "uintptr_t",
})


def _re_c_name(name: str) -> str:
    """Best-effort C name for records serialized without ``c_name``."""
    if name.startswith(("Vk", "VK_", "PFN_", "StdVideo")):
        return name
    if name and name[0].islower():
        return "vk" + name[0].upper() + name[1:]
    return "Vk" + name


def _re_c_type(name: str) -> str:
    """Best-effort C type spelling for a processed type name."""
    if not name or name.startswith(("Vk", "VK_", "PFN_", "StdVideo")):
        return name
    if name in _PRIMITIVE_C_TYPES or "*" in name:
        return name
    return "Vk" + name


# ---------------------------------------------------------------------------
# Small shared records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Availability:
    """Feature/extension availability plus preprocessor guard of an entity.

    ``doc`` carries the comment attribute of the feature/extension/require
    block that introduced the entity, so requirement-level documentation is
    retained even for entities that have no comment of their own.
    """

    features: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    protect: str | None = None
    doc: str | None = None

    def merge(self, other: "Availability") -> "Availability":
        return Availability(
            tuple(dict.fromkeys((*self.features, *other.features))),
            tuple(dict.fromkeys((*self.extensions, *other.extensions))),
            self.protect or other.protect,
            self.doc or other.doc,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "extensions": list(self.extensions),
            "protect": self.protect,
            "doc": self.doc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Availability":
        return cls(
            tuple(data.get("features", ())),
            tuple(data.get("extensions", ())),
            data.get("protect"),
            data.get("doc"),
        )


@dataclass(frozen=True, slots=True)
class Length:
    """One ``len`` entry from the XML.

    ``text`` is the raw attribute value.  ``latex`` carries the body of
    ``latexmath:[...]`` entries when present (e.g. ``codeSize \\over 4``).
    """

    text: str
    latex: str | None = None

    @property
    def identifiers(self) -> tuple[str, ...]:
        """Identifier-like tokens referenced by this length expression.

        LaTeX control words (``textrm``, ``over``, ...) are filtered out so
        only real value references remain.
        """
        import re

        latex_controls = {
            "textrm", "mathrm", "mathit", "operatorname", "text", "over",
            "frac", "left", "right", "lceil", "rceil", "lfloor", "rfloor",
            "lvert", "rvert", "lVert", "rVert", "big", "bigg", "Big",
            "Bigg", "binom", "qquad", "quad", "cdot", "times", "div",
        }
        tokens = re.findall(r"[A-Za-z_]\w*", self.latex or self.text)
        return tuple(
            dict.fromkeys(token for token in tokens if token not in latex_controls)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "latex": self.latex}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Length":
        return cls(data["text"], data.get("latex"))


# ---------------------------------------------------------------------------
# Parameters / struct members
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Param:
    """A command parameter or a struct member with processed metadata.

    ``name`` is the C spelling (``pCreateInfo``); :attr:`public_name` is the
    normalized name with Vulkan's pointer prefixes removed (``createInfo``).
    ``type`` is the processed type name (``TensorViewARM``, ``uint32_t``) while
    ``c_type`` keeps the original C type name (``VkTensorViewARM``).

    The C declarator is never stored verbatim.  It is reconstructed from the
    processed fields via :attr:`c_declaration`: leading ``const`` + ``c_type``
    + per-level pointer constness (:attr:`pointer_consts`) + ``name`` +
    :attr:`c_suffix` (array brackets), e.g. ``const VkBufferCreateInfo*
    pCreateInfo``, ``const char* const* ppEnabledLayerNames``, or
    ``const float blendConstants[4]``.
    """

    name: str
    type: str
    c_type: str
    c_suffix: str = ""
    pointer_depth: int = 0
    pointer_consts: tuple[bool, ...] = ()
    const: bool = False
    optional: tuple[str, ...] = ()
    lengths: tuple[Length, ...] = ()
    alt_length: str | None = None
    externsync: str | None = None
    selector: str | None = None
    selection: str | None = None
    values: str | None = None
    object_type: str | None = None
    no_auto_validity: bool = False
    doc: str | None = None
    # Computed during IR construction:
    direction: str = "input"           # "input" | "output"
    counts_for: tuple[str, ...] = ()   # array params sized by this count

    @property
    def public_name(self) -> str:
        """C++-facing name: leading p/pp Vulkan prefixes removed."""
        import re

        if self.pointer_depth and re.match(r"p+[A-Z]", self.name):
            base = re.sub(r"^p+(?=[A-Z])", "", self.name)
            return base[:1].lower() + base[1:]
        return self.name

    @property
    def is_optional(self) -> bool:
        return "true" in self.optional

    @property
    def is_count(self) -> bool:
        return bool(self.counts_for)

    @property
    def is_array(self) -> bool:
        return (self.pointer_depth >= 1 and bool(self.lengths)) or bool(self.c_suffix)

    @property
    def is_byte_array(self) -> bool:
        """True when XML ``altlen`` says the length counts bytes."""
        return self.alt_length is not None

    @property
    def length_names(self) -> tuple[str, ...]:
        """Names referenced by the length expressions (deduplicated)."""
        return tuple(
            dict.fromkeys(
                name
                for length in self.lengths
                for name in length.identifiers
            )
        )

    @property
    def c_declaration(self) -> str:
        """The exact C declarator, reconstructed from processed fields."""
        prefix = "const " if self.const else ""
        stars = "".join(
            "* const" if level_const else "*"
            for level_const in self.pointer_consts
        )
        return f"{prefix}{self.c_type}{stars} {self.name}{self.c_suffix}"

    @property
    def c_signature_piece(self) -> str:
        return self.c_declaration

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "c_type": self.c_type,
            "c_suffix": self.c_suffix,
            "pointer_depth": self.pointer_depth,
            "pointer_consts": list(self.pointer_consts),
            "const": self.const,
            "optional": list(self.optional),
            "lengths": [length.to_dict() for length in self.lengths],
            "alt_length": self.alt_length,
            "externsync": self.externsync,
            "selector": self.selector,
            "selection": self.selection,
            "values": self.values,
            "object_type": self.object_type,
            "no_auto_validity": self.no_auto_validity,
            "doc": self.doc,
            "direction": self.direction,
            "counts_for": list(self.counts_for),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Param":
        return cls(
            name=data["name"],
            type=data["type"],
            c_type=data["c_type"],
            c_suffix=data.get("c_suffix", ""),
            pointer_depth=int(data.get("pointer_depth", 0)),
            pointer_consts=tuple(bool(item) for item in data.get("pointer_consts", ())),
            const=bool(data.get("const", False)),
            optional=tuple(data.get("optional", ())),
            lengths=tuple(Length.from_dict(item) for item in data.get("lengths", ())),
            alt_length=data.get("alt_length"),
            externsync=data.get("externsync"),
            selector=data.get("selector"),
            selection=data.get("selection"),
            values=data.get("values"),
            object_type=data.get("object_type"),
            no_auto_validity=bool(data.get("no_auto_validity", False)),
            doc=data.get("doc"),
            direction=data.get("direction", "input"),
            counts_for=tuple(data.get("counts_for", ())),
        )


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Handle:
    """A Vulkan handle type (``VkBuffer``, ``VkInstance``, ...).

    ``name`` is the processed general name (``Buffer``); ``c_name`` keeps the
    C API spelling (``VkBuffer``).
    """

    name: str
    c_name: str
    parents: tuple[str, ...] = ()
    dispatchable: bool = False
    object_type_enum: str | None = None
    doc: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)
    active: bool = True
    # Computed relationships:
    create_infos: tuple[str, ...] = ()   # create-info types accepted by producers
    create_info: str | None = None       # concrete record type (or synthesized name)
    releaser: str | None = None          # vkDestroy*/vkFree*/vkRelease* command name

    @property
    def parent(self) -> str | None:
        return self.parents[0] if self.parents else None

    @property
    def c_declaration(self) -> str:
        """Handle macro invocation reconstructed from dispatchability."""
        macro = "VK_DEFINE_HANDLE" if self.dispatchable else "VK_DEFINE_NON_DISPATCHABLE_HANDLE"
        return f"{macro}({self.c_name})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_name": self.c_name,
            "parents": list(self.parents),
            "dispatchable": self.dispatchable,
            "object_type_enum": self.object_type_enum,
            "doc": self.doc,
            "protect": self.protect,
            "availability": self.availability.to_dict(),
            "active": self.active,
            "create_infos": list(self.create_infos),
            "create_info": self.create_info,
            "releaser": self.releaser,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Handle":
        return cls(
            name=data["name"],
            c_name=data.get("c_name") or _re_c_name(data["name"]),
            parents=tuple(data.get("parents", ())),
            dispatchable=bool(data.get("dispatchable", False)),
            object_type_enum=data.get("object_type_enum"),
            doc=data.get("doc"),
            protect=data.get("protect"),
            availability=Availability.from_dict(data.get("availability", {})),
            active=bool(data.get("active", True)),
            create_infos=tuple(data.get("create_infos", ())),
            create_info=data.get("create_info"),
            releaser=data.get("releaser"),
        )


@dataclass(slots=True)
class Struct:
    """A struct or union declaration with processed members."""

    name: str
    c_name: str
    category: str  # "struct" | "union"
    members: tuple[Param, ...] = ()
    struct_extends: tuple[str, ...] = ()
    returned_only: bool = False
    allow_duplicate: bool = False
    doc: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)
    active: bool = True

    @property
    def c_declaration(self) -> str:
        """C typedef text reconstructed from the processed members."""
        members = "; ".join(member.c_declaration for member in self.members)
        if members:
            members += ";"
        keyword = "union" if self.category == "union" else "struct"
        return f"typedef {keyword} {self.c_name} {{ {members} }} {self.c_name};".replace("  ", " ")

    def member(self, name: str) -> Param | None:
        return next((member for member in self.members if member.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_name": self.c_name,
            "category": self.category,
            "members": [member.to_dict() for member in self.members],
            "struct_extends": list(self.struct_extends),
            "returned_only": self.returned_only,
            "allow_duplicate": self.allow_duplicate,
            "doc": self.doc,
            "protect": self.protect,
            "availability": self.availability.to_dict(),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Struct":
        return cls(
            name=data["name"],
            c_name=data.get("c_name") or _re_c_name(data["name"]),
            category=data.get("category", "struct"),
            members=tuple(Param.from_dict(item) for item in data.get("members", ())),
            struct_extends=tuple(data.get("struct_extends", ())),
            returned_only=bool(data.get("returned_only", False)),
            allow_duplicate=bool(data.get("allow_duplicate", False)),
            doc=data.get("doc"),
            protect=data.get("protect"),
            availability=Availability.from_dict(data.get("availability", {})),
            active=bool(data.get("active", True)),
        )


@dataclass(slots=True)
class EnumValue:
    """One enumerator of an enum/bitmask group."""

    name: str
    value: str | None = None
    bitpos: int | None = None
    offset: int | None = None
    extnumber: int | None = None
    negative: bool = False
    alias_of: str | None = None
    doc: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "bitpos": self.bitpos,
            "offset": self.offset,
            "extnumber": self.extnumber,
            "negative": self.negative,
            "alias_of": self.alias_of,
            "doc": self.doc,
            "protect": self.protect,
            "availability": self.availability.to_dict(),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnumValue":
        return cls(
            name=data["name"],
            value=data.get("value"),
            bitpos=data.get("bitpos"),
            offset=data.get("offset"),
            extnumber=data.get("extnumber"),
            negative=bool(data.get("negative", False)),
            alias_of=data.get("alias_of"),
            doc=data.get("doc"),
            protect=data.get("protect"),
            availability=Availability.from_dict(data.get("availability", {})),
            active=bool(data.get("active", True)),
        )


@dataclass(slots=True)
class Enum:
    """A named enum group (``VkResult``, ``VkBufferUsageFlagBits``, ...).

    Values keep their C spelling (``VK_SUCCESS``); only the group name is
    processed into ``name`` with the C name kept in ``c_name``.
    """

    name: str
    c_name: str
    kind: str  # "enum" | "bitmask" | "constants"
    bitwidth: int | None = None
    values: tuple[EnumValue, ...] = ()
    doc: str | None = None
    availability: Availability = field(default_factory=Availability)

    @property
    def is_bitmask(self) -> bool:
        return self.kind == "bitmask"

    def value(self, name: str) -> EnumValue | None:
        return next((item for item in self.values if item.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_name": self.c_name,
            "kind": self.kind,
            "bitwidth": self.bitwidth,
            "values": [value.to_dict() for value in self.values],
            "doc": self.doc,
            "availability": self.availability.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Enum":
        return cls(
            name=data["name"],
            c_name=data.get("c_name") or _re_c_name(data["name"]),
            kind=data.get("kind", "enum"),
            bitwidth=data.get("bitwidth"),
            values=tuple(EnumValue.from_dict(item) for item in data.get("values", ())),
            doc=data.get("doc"),
            availability=Availability.from_dict(data.get("availability", {})),
        )


@dataclass(slots=True)
class Bitmask:
    """A bitmask typedef (``VkBufferUsageFlags``) plus its bit domain.

    ``base`` and ``bits`` reference other entities by their processed general
    names (``Flags``, ``BufferUsageFlagBits``); :attr:`c_declaration` maps
    them back to C spellings.
    """

    name: str
    c_name: str
    base: str | None = None          # underlying mask type (Flags/Flags64)
    bits: str | None = None          # enum group name (BufferUsageFlagBits)
    alias_of: str | None = None
    doc: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)
    active: bool = True

    @property
    def c_declaration(self) -> str:
        """C typedef reconstructed from the stored base type."""
        return f"typedef {_re_c_type(self.base)} {self.c_name};" if self.base else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_name": self.c_name,
            "base": self.base,
            "bits": self.bits,
            "alias_of": self.alias_of,
            "doc": self.doc,
            "protect": self.protect,
            "availability": self.availability.to_dict(),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Bitmask":
        return cls(
            name=data["name"],
            c_name=data.get("c_name") or _re_c_name(data["name"]),
            base=data.get("base"),
            bits=data.get("bits"),
            alias_of=data.get("alias_of"),
            doc=data.get("doc"),
            protect=data.get("protect"),
            availability=Availability.from_dict(data.get("availability", {})),
            active=bool(data.get("active", True)),
        )


@dataclass(slots=True)
class Basetype:
    """A base type typedef (``VkBool32``, ``VkDeviceSize``, ``VkFlags``...).

    ``base`` is a C primitive spelling (``uint32_t``), not an entity
    reference, so it stays verbatim.
    """

    name: str
    c_name: str
    base: str | None = None          # underlying C type (uint32_t, uint64_t, ...)
    doc: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)

    @property
    def c_declaration(self) -> str:
        """C typedef reconstructed from the stored base type."""
        return f"typedef {self.base} {self.c_name};" if self.base else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_name": self.c_name,
            "base": self.base,
            "doc": self.doc,
            "protect": self.protect,
            "availability": self.availability.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Basetype":
        return cls(
            name=data["name"],
            c_name=data.get("c_name") or _re_c_name(data["name"]),
            base=data.get("base"),
            doc=data.get("doc"),
            protect=data.get("protect"),
            availability=Availability.from_dict(data.get("availability", {})),
        )


@dataclass(slots=True)
class FuncPointer:
    """A function-pointer typedef (``PFN_vkVoidFunction``, ...).

    The declaration is parsed into a return type and processed parameters;
    the C typedef text is reconstructed via :attr:`c_declaration`.  ``PFN_``
    names carry no Vulkan prefix, so ``name`` equals ``c_name`` for them.
    """

    name: str
    c_name: str
    return_type: str | None = None   # processed (void, Bool32, ...)
    c_return_type: str | None = None  # C spelling (void, VkBool32, ...)
    params: tuple[Param, ...] = ()
    doc: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)

    @property
    def c_declaration(self) -> str:
        """C typedef reconstructed from return type and processed params."""
        if not self.c_return_type:
            return ""
        arguments = ", ".join(param.c_declaration for param in self.params)
        if not arguments:
            arguments = "void"
        return f"typedef {self.c_return_type} (VKAPI_PTR *{self.c_name})({arguments});"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_name": self.c_name,
            "return_type": self.return_type,
            "c_return_type": self.c_return_type,
            "params": [param.to_dict() for param in self.params],
            "doc": self.doc,
            "protect": self.protect,
            "availability": self.availability.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FuncPointer":
        return cls(
            name=data["name"],
            c_name=data.get("c_name") or _re_c_name(data["name"]),
            return_type=data.get("return_type"),
            c_return_type=data.get("c_return_type") or _re_c_type(data.get("return_type") or ""),
            params=tuple(Param.from_dict(item) for item in data.get("params", ())),
            doc=data.get("doc"),
            protect=data.get("protect"),
            availability=Availability.from_dict(data.get("availability", {})),
        )


@dataclass(slots=True)
class Alias:
    """A named type alias (``VkBufferEXT`` -> ``VkBuffer``).

    ``target`` references the aliased entity by its processed general name
    (``Buffer``).
    """

    name: str
    c_name: str
    target: str
    resolved_category: str | None = None  # category of the resolved target
    doc: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_name": self.c_name,
            "target": self.target,
            "resolved_category": self.resolved_category,
            "doc": self.doc,
            "protect": self.protect,
            "availability": self.availability.to_dict(),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Alias":
        return cls(
            name=data["name"],
            c_name=data.get("c_name") or _re_c_name(data["name"]),
            target=data["target"],
            resolved_category=data.get("resolved_category"),
            doc=data.get("doc"),
            protect=data.get("protect"),
            availability=Availability.from_dict(data.get("availability", {})),
            active=bool(data.get("active", True)),
        )


@dataclass(slots=True)
class Define:
    """A ``category="define"`` macro (``VK_DEFINE_HANDLE``, ...).

    ``body`` is the macro expansion after the name; ``disabled`` marks the few
    registries entries that are commented out in the XML.  Macro names carry
    the ``VK_`` prefix, so ``name`` equals ``c_name``.
    """

    name: str
    c_name: str
    body: str = ""
    requires: str | None = None
    disabled: bool = False
    doc: str | None = None
    protect: str | None = None

    @property
    def c_declaration(self) -> str:
        """``#define`` line reconstructed from the stored macro body."""
        prefix = "//" if self.disabled else ""
        return f"{prefix}#define {self.c_name} {self.body}".rstrip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_name": self.c_name,
            "body": self.body,
            "requires": self.requires,
            "disabled": self.disabled,
            "doc": self.doc,
            "protect": self.protect,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Define":
        return cls(
            name=data["name"],
            c_name=data.get("c_name") or data["name"],
            body=data.get("body", ""),
            requires=data.get("requires"),
            disabled=bool(data.get("disabled", False)),
            doc=data.get("doc"),
            protect=data.get("protect"),
        )


@dataclass(slots=True)
class RawType:
    """A type entry without a modeled category.

    Foreign/platform tokens (``Display``, ``HWND``, ``wl_display``, ...),
    supplemental fixed-width integers and ``include`` records land here so
    that no input declaration is lost.  ``target`` carries the parsed include
    header path when the record is an ``include``.
    """

    name: str
    c_name: str
    category: str | None = None  # raw XML category when present
    target: str | None = None    # include header path, when applicable
    alias_of: str | None = None
    doc: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)

    @property
    def c_declaration(self) -> str:
        """Include line reconstructed from the stored target."""
        if self.category == "include" and self.target:
            return f'#include "{self.target}"'
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_name": self.c_name,
            "category": self.category,
            "target": self.target,
            "alias_of": self.alias_of,
            "doc": self.doc,
            "protect": self.protect,
            "availability": self.availability.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawType":
        return cls(
            name=data["name"],
            c_name=data.get("c_name") or data["name"],
            category=data.get("category"),
            target=data.get("target"),
            alias_of=data.get("alias_of"),
            doc=data.get("doc"),
            protect=data.get("protect"),
            availability=Availability.from_dict(data.get("availability", {})),
        )


@dataclass(slots=True)
class Constant:
    """A named constant (``VK_HEADER_VERSION``, ``VK_WHOLE_SIZE``, ...).

    Constant names carry the ``VK_`` prefix, so ``name`` equals ``c_name``.
    """

    name: str
    c_name: str
    value: str | None = None
    alias_of: str | None = None
    c_type: str | None = None
    doc: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_name": self.c_name,
            "value": self.value,
            "alias_of": self.alias_of,
            "c_type": self.c_type,
            "doc": self.doc,
            "protect": self.protect,
            "availability": self.availability.to_dict(),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Constant":
        return cls(
            name=data["name"],
            c_name=data.get("c_name") or data["name"],
            value=data.get("value"),
            alias_of=data.get("alias_of"),
            c_type=data.get("c_type"),
            doc=data.get("doc"),
            protect=data.get("protect"),
            availability=Availability.from_dict(data.get("availability", {})),
            active=bool(data.get("active", True)),
        )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Command:
    """A Vulkan command with processed parameters and relationships.

    ``name`` is the processed general name (``createBuffer``); ``c_name``
    keeps the C API name (``vkCreateBuffer``).  ``return_type`` is the
    processed type name (``Result``) with ``c_return_type`` keeping the C
    spelling (``VkResult``).  The C signature is never stored:
    :attr:`c_signature` reconstructs it from those fields and the processed
    parameters.  ``receivers`` / ``member_names`` record who holds the
    command as a member function and under which receiver-relative name,
    using general handle names.
    """

    name: str
    c_name: str
    return_type: str
    c_return_type: str
    params: tuple[Param, ...] = ()
    alias_of: str | None = None
    export: tuple[str, ...] = ()
    success_codes: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    queues: tuple[str, ...] = ()
    renderpass: str | None = None
    command_buffer_levels: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    implicit_externsync: tuple[str, ...] = ()
    doc: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)
    active: bool = True
    # Computed during IR construction:
    cpp_name: str = ""                          # receiver-independent C++ name
    dispatch: str | None = None                 # first handle param (VkDevice/...)
    receivers: tuple[str, ...] = ()             # handle types hosting this command
    member_names: dict[str, str] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()               # writable output param names
    count_param: str | None = None              # two-call enumeration count param
    vector_output: str | None = None            # two-call enumeration vector param
    status_alternatives: bool = False           # more than one success status
    count_name: str | None = None               # normalized public count name
    owned_outputs: tuple[str, ...] = ()         # outputs whose wrapper owns the object

    def param(self, name: str) -> Param | None:
        return next((param for param in self.params if param.name == name), None)

    @property
    def c_signature(self) -> str:
        """Exact C API signature, e.g. ``VkResult vkCreateBuffer(VkDevice device,
        const VkBufferCreateInfo* pCreateInfo, ..., VkBuffer* pBuffer)``."""
        pieces = ", ".join(param.c_signature_piece for param in self.params)
        return f"{self.c_return_type} {self.c_name}({pieces})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_name": self.c_name,
            "return_type": self.return_type,
            "c_return_type": self.c_return_type,
            "params": [param.to_dict() for param in self.params],
            "alias_of": self.alias_of,
            "export": list(self.export),
            "success_codes": list(self.success_codes),
            "error_codes": list(self.error_codes),
            "queues": list(self.queues),
            "renderpass": self.renderpass,
            "command_buffer_levels": list(self.command_buffer_levels),
            "tasks": list(self.tasks),
            "implicit_externsync": list(self.implicit_externsync),
            "doc": self.doc,
            "protect": self.protect,
            "availability": self.availability.to_dict(),
            "active": self.active,
            "cpp_name": self.cpp_name,
            "dispatch": self.dispatch,
            "receivers": list(self.receivers),
            "member_names": dict(self.member_names),
            "outputs": list(self.outputs),
            "count_param": self.count_param,
            "vector_output": self.vector_output,
            "status_alternatives": self.status_alternatives,
            "count_name": self.count_name,
            "owned_outputs": list(self.owned_outputs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Command":
        return cls(
            name=data["name"],
            c_name=data.get("c_name") or _re_c_name(data["name"]),
            return_type=data.get("return_type", ""),
            c_return_type=data.get("c_return_type") or _re_c_type(data.get("return_type", "")),
            params=tuple(Param.from_dict(item) for item in data.get("params", ())),
            alias_of=data.get("alias_of"),
            export=tuple(data.get("export", ())),
            success_codes=tuple(data.get("success_codes", ())),
            error_codes=tuple(data.get("error_codes", ())),
            queues=tuple(data.get("queues", ())),
            renderpass=data.get("renderpass"),
            command_buffer_levels=tuple(data.get("command_buffer_levels", ())),
            tasks=tuple(data.get("tasks", ())),
            implicit_externsync=tuple(data.get("implicit_externsync", ())),
            doc=data.get("doc"),
            protect=data.get("protect"),
            availability=Availability.from_dict(data.get("availability", {})),
            active=bool(data.get("active", True)),
            cpp_name=data.get("cpp_name", ""),
            dispatch=data.get("dispatch"),
            receivers=tuple(data.get("receivers", ())),
            member_names=dict(data.get("member_names", {})),
            outputs=tuple(data.get("outputs", ())),
            count_param=data.get("count_param"),
            vector_output=data.get("vector_output"),
            status_alternatives=bool(data.get("status_alternatives", False)),
            count_name=data.get("count_name"),
            owned_outputs=tuple(data.get("owned_outputs", ())),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IrRegistry:
    """The complete processed middle layer for one or more registry files."""

    sources: tuple[Path, ...] = ()
    api: str = "vulkan"
    tags: tuple[str, ...] = ()
    platforms: dict[str, str] = field(default_factory=dict)
    header_version: str | None = None
    handles: dict[str, Handle] = field(default_factory=dict)
    structs: dict[str, Struct] = field(default_factory=dict)
    enums: dict[str, Enum] = field(default_factory=dict)
    bitmasks: dict[str, Bitmask] = field(default_factory=dict)
    basetypes: dict[str, Basetype] = field(default_factory=dict)
    func_pointers: dict[str, FuncPointer] = field(default_factory=dict)
    aliases: dict[str, Alias] = field(default_factory=dict)
    defines: dict[str, Define] = field(default_factory=dict)
    raw_types: dict[str, RawType] = field(default_factory=dict)
    constants: dict[str, Constant] = field(default_factory=dict)
    commands: dict[str, Command] = field(default_factory=dict)

    # -- lookups -------------------------------------------------------------

    def type_category(self, name: str) -> str | None:
        """Resolved category of a type name (alias targets included)."""
        if name in self.handles:
            return "handle"
        if name in self.structs:
            return self.structs[name].category
        if name in self.enums:
            return "enum"
        if name in self.bitmasks:
            return "bitmask"
        if name in self.basetypes:
            return "basetype"
        if name in self.func_pointers:
            return "funcpointer"
        if name in self.raw_types:
            return self.raw_types[name].category or "opaque"
        alias = self.aliases.get(name)
        if alias is not None:
            return alias.resolved_category or self.type_category(alias.target)
        return None

    def resolve(self, name: str) -> Handle | Struct | Enum | Bitmask | Basetype | FuncPointer | RawType | Alias | None:
        """Resolve a type name to its IR entity (following aliases)."""
        seen: set[str] = set()
        current = name
        while current not in seen:
            seen.add(current)
            for collection in (
                self.handles,
                self.structs,
                self.enums,
                self.bitmasks,
                self.basetypes,
                self.func_pointers,
                self.raw_types,
            ):
                if current in collection:
                    return collection[current]
            alias = self.aliases.get(current)
            if alias is None:
                return None
            current = alias.target
        return None

    @property
    def handle_names(self) -> frozenset[str]:
        """Canonical handle type names plus aliases that resolve to handles."""
        names = set(self.handles)
        names.update(
            name for name, alias in self.aliases.items() if alias.resolved_category == "handle"
        )
        return frozenset(names)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [str(source) for source in self.sources],
            "api": self.api,
            "tags": list(self.tags),
            "platforms": dict(self.platforms),
            "header_version": self.header_version,
            "handles": {name: value.to_dict() for name, value in self.handles.items()},
            "structs": {name: value.to_dict() for name, value in self.structs.items()},
            "enums": {name: value.to_dict() for name, value in self.enums.items()},
            "bitmasks": {name: value.to_dict() for name, value in self.bitmasks.items()},
            "basetypes": {name: value.to_dict() for name, value in self.basetypes.items()},
            "func_pointers": {name: value.to_dict() for name, value in self.func_pointers.items()},
            "aliases": {name: value.to_dict() for name, value in self.aliases.items()},
            "defines": {name: value.to_dict() for name, value in self.defines.items()},
            "raw_types": {name: value.to_dict() for name, value in self.raw_types.items()},
            "constants": {name: value.to_dict() for name, value in self.constants.items()},
            "commands": {name: value.to_dict() for name, value in self.commands.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IrRegistry":
        return cls(
            sources=tuple(Path(item) for item in data.get("sources", ())),
            api=data.get("api", "vulkan"),
            tags=tuple(data.get("tags", ())),
            platforms=dict(data.get("platforms", {})),
            header_version=data.get("header_version"),
            handles={name: Handle.from_dict(value) for name, value in data.get("handles", {}).items()},
            structs={name: Struct.from_dict(value) for name, value in data.get("structs", {}).items()},
            enums={name: Enum.from_dict(value) for name, value in data.get("enums", {}).items()},
            bitmasks={name: Bitmask.from_dict(value) for name, value in data.get("bitmasks", {}).items()},
            basetypes={name: Basetype.from_dict(value) for name, value in data.get("basetypes", {}).items()},
            func_pointers={name: FuncPointer.from_dict(value) for name, value in data.get("func_pointers", {}).items()},
            aliases={name: Alias.from_dict(value) for name, value in data.get("aliases", {}).items()},
            defines={name: Define.from_dict(value) for name, value in data.get("defines", {}).items()},
            raw_types={name: RawType.from_dict(value) for name, value in data.get("raw_types", {}).items()},
            constants={name: Constant.from_dict(value) for name, value in data.get("constants", {}).items()},
            commands={name: Command.from_dict(value) for name, value in data.get("commands", {}).items()},
        )

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "IrRegistry":
        return cls.from_dict(json.loads(text))
