from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Availability:
    features: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    protect: str | None = None

    def merge(self, other: "Availability") -> "Availability":
        return Availability(
            tuple(dict.fromkeys((*self.features, *other.features))),
            tuple(dict.fromkeys((*self.extensions, *other.extensions))),
            self.protect or other.protect,
        )


@dataclass(slots=True)
class Member:
    name: str
    type: str
    declaration: str
    pointer_depth: int = 0
    const: bool = False
    optional: tuple[str, ...] = ()
    length: tuple[str, ...] = ()
    alt_length: str | None = None
    externsync: bool = False
    externsync_expression: str | None = None
    values: str | None = None
    selector: str | None = None
    selection: str | None = None
    object_type: str | None = None
    no_auto_validity: bool = False


@dataclass(slots=True)
class TypeDecl:
    name: str
    category: str | None
    declaration: str
    alias: str | None = None
    parent: str | None = None
    object_type_enum: str | None = None
    struct_extends: tuple[str, ...] = ()
    returned_only: bool = False
    allow_duplicate: bool = False
    protect: str | None = None
    requires: str | None = None
    members: list[Member] = field(default_factory=list)
    availability: Availability = field(default_factory=Availability)


@dataclass(slots=True)
class EnumValue:
    name: str
    value: str | None = None
    bitpos: int | None = None
    offset: int | None = None
    extnumber: int | None = None
    negative: bool = False
    alias: str | None = None
    comment: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)


@dataclass(slots=True)
class EnumGroup:
    name: str
    kind: str
    bitwidth: int | None = None
    values: list[EnumValue] = field(default_factory=list)
    availability: Availability = field(default_factory=Availability)


@dataclass(slots=True)
class Command:
    name: str
    return_type: str
    declaration: str
    params: list[Member] = field(default_factory=list)
    alias: str | None = None
    success_codes: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    queues: tuple[str, ...] = ()
    renderpass: str | None = None
    command_buffer_levels: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    implicit_externsync: tuple[str, ...] = ()
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)


@dataclass(slots=True)
class Constant:
    name: str
    value: str | None = None
    alias: str | None = None
    type: str | None = None
    protect: str | None = None
    availability: Availability = field(default_factory=Availability)


@dataclass(slots=True)
class Registry:
    sources: tuple[Path, ...]
    types: dict[str, TypeDecl] = field(default_factory=dict)
    enums: dict[str, EnumGroup] = field(default_factory=dict)
    commands: dict[str, Command] = field(default_factory=dict)
    constants: dict[str, Constant] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    platforms: dict[str, str] = field(default_factory=dict)

    @property
    def handles(self) -> list[TypeDecl]:
        return [value for value in self.types.values() if value.category == "handle"]

    @property
    def structs(self) -> list[TypeDecl]:
        return [value for value in self.types.values() if value.category in {"struct", "union"}]
