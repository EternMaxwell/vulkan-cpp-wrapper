"""Build the processed IR (:mod:`vulkan_wrapper_gen.ir.model`) from Khronos
registry XML files.

The builder performs all registry-specific normalization:

* merges multiple registry files (vk.xml + video.xml + ...), preferring the
  complete declaration when a supplemental registry upgrades a placeholder,
* resolves type aliases and records each alias's resolved category,
* links every ``len``/``altlen`` reference to the count parameter it sizes and
  vice versa (:attr:`Param.counts_for`),
* classifies each parameter/member as input or output,
* computes command dispatch handle, receivers, receiver-relative member names,
  two-call enumeration shapes, alternative success statuses, owned outputs,
  per-handle creation records and releasers,
* keeps availability, platform guards, docs/comments, and every raw XML
  attribute.

Nothing is dropped: entities that no active feature/extension requires remain
in the IR with ``active=False``.
"""

from __future__ import annotations

from copy import deepcopy
import fnmatch
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from ..config import GeneratorConfig
from ..naming import strip_vk, strip_vk_command
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


class RegistryError(ValueError):
    """Raised when registry XML input is missing, malformed, or conflicting."""


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())


def _generalize(name: str) -> str:
    """C name -> processed general name (the IR dict key).

    Commands drop the lowercase ``vk`` prefix and decapitalize
    (``vkCreateBuffer`` -> ``createBuffer``); types drop the ``Vk`` prefix
    (``VkBuffer`` -> ``Buffer``).  ``VK_``/``PFN_``/``StdVideo`` names and
    foreign tokens are kept verbatim.
    """
    if name.startswith("vk") and len(name) > 2 and name[2].isupper():
        return strip_vk_command(name)
    return strip_vk(name)


def _applies(element: ET.Element, api: str) -> bool:
    declared = _csv(element.get("api"))
    return not declared or api in declared or f"{api}base" in declared


def _split_comments(element: ET.Element) -> tuple[str, str | None]:
    """Return (declaration text without comments, joined comment text).

    Only *direct* ``<comment>`` children (plus the ``comment`` attribute)
    describe the entity itself; nested comments belong to members/parameters
    and are captured by their own ``_param`` call.
    """
    comments = [
        (item.text or "").strip()
        for item in element
        if item.tag == "comment" and (item.text or "").strip()
    ]
    clone = deepcopy(element)
    for comment in clone.findall(".//comment"):
        parent = next((candidate for candidate in clone.iter() if comment in list(candidate)), None)
        if parent is not None:
            parent.remove(comment)
    text = re.sub(r"\s+", " ", "".join(clone.itertext())).strip()
    doc = " ".join(comments) if comments else None
    if element.get("comment"):
        attribute = element.get("comment", "").strip()
        doc = f"{doc} {attribute}".strip() if doc else attribute
    return text, doc


def _text(element: ET.Element) -> str:
    text, _ = _split_comments(element)
    # Collapse XML whitespace/newlines so declarators read as plain C.
    return re.sub(r"\s+", " ", text).strip()


def _pointer_depth(declaration: str, name: str) -> int:
    prefix = declaration.rsplit(name, 1)[0] if name in declaration else declaration
    return prefix.count("*")


def _latex_body(value: str) -> str | None:
    marker = "latexmath:["
    if not value.startswith(marker) or not value.endswith("]"):
        return None
    return value[len(marker):-1]


def _param(element: ET.Element, handle_names: frozenset[str]) -> Param:
    name = element.findtext("name")
    type_name = element.findtext("type")
    if not name or not type_name:
        raise RegistryError(f"member/parameter is missing a name or type: {_text(element)!r}")
    declaration, doc = _split_comments(element)
    declaration = re.sub(r"\s+", " ", declaration).strip()
    prefix = declaration.rsplit(name, 1)[0] if name in declaration else declaration
    suffix = declaration.rsplit(name, 1)[1] if name in declaration else ""
    lengths = tuple(
        Length(value, _latex_body(value))
        for value in _csv(element.get("len"))
    )
    alt_length = element.get("altlen")
    # Per-pointer-level constness.  The leading const covers the pointee;
    # each following `const` belongs to the star that precedes it, e.g.
    # `const char* const*` -> const=True, pointer_consts=(True, False).
    depth = _pointer_depth(declaration, name)
    remainder = re.sub(r"^const\s+", "", prefix)
    segments = remainder.split("*")
    pointer_consts = tuple(
        "const" in segments[index + 1].split()
        for index in range(depth)
    ) if depth else ()
    param = Param(
        name=name,
        type=strip_vk(type_name),
        c_type=type_name,
        c_suffix=suffix,
        pointer_depth=depth,
        pointer_consts=pointer_consts,
        const=bool(re.search(r"\bconst\b", prefix)),
        optional=_csv(element.get("optional")),
        lengths=lengths,
        alt_length=alt_length,
        externsync=element.get("externsync"),
        selector=element.get("selector"),
        selection=element.get("selection"),
        values=element.get("values"),
        object_type=element.get("objecttype"),
        no_auto_validity=element.get("noautovalidity") == "true",
        doc=doc,
    )
    param.direction = _direction(param, handle_names)
    return param


def _direction(param: Param, handle_names: frozenset[str]) -> str:
    """Classify a parameter/member as input (readonly) or output (write).

    Vulkan expresses writable storage as non-const pointers.  Two conventions
    escape the simple const rule: ``pUserData`` is an opaque input pointer that
    the C headers cannot spell as const, and platform APIs such as Xlib pass
    mutable foreign pointers (``Display*``) as input.  By-value parameters and
    const pointers are always input.
    """
    if param.pointer_depth == 0 or param.const:
        return "input"
    if param.c_type == "void" and param.pointer_depth == 1 and param.name == "pUserData":
        return "input"
    if param.name.startswith("p") or param.type in handle_names:
        return "output"
    return "input"


def _link_counts(params: list[Param]) -> None:
    """Wire length references to the count parameter they size and back."""
    by_name = {param.name: param for param in params}
    for param in params:
        matched: list[str] = []
        for identifier in param.length_names:
            target = identifier if identifier in by_name else None
            if target is None and identifier.startswith("p") and len(identifier) > 1:
                stripped = identifier[1:]
                if stripped in by_name:
                    target = stripped
            if target is not None and target not in matched:
                matched.append(target)
        for target in matched:
            count = by_name[target]
            if param.name not in count.counts_for:
                count.counts_for += (param.name,)


def _is_complete(value: object) -> bool:
    """Whether a merged entity carries real content (not a placeholder)."""
    if isinstance(value, Struct):
        return bool(value.members)
    if isinstance(value, Handle):
        return bool(value.object_type_enum or value.parents or value.dispatchable)
    if isinstance(value, Command):
        return bool(value.params)
    if isinstance(value, Define):
        return bool(value.body)
    if isinstance(value, (Bitmask, Basetype)):
        return bool(value.base)
    if isinstance(value, FuncPointer):
        return bool(value.return_type)
    if isinstance(value, RawType):
        return bool(value.target)
    return True


def _add_unique(target: dict[str, object], key: str, value: object, source: Path) -> None:
    """Merge declarations across registries; the complete one wins."""
    old = target.get(key)
    if old is None:
        target[key] = value
        return
    if old == value:
        return
    if not _is_complete(old) and _is_complete(value):
        target[key] = value
        return
    if _is_complete(old) and _is_complete(value):
        # Supplemental registries repeat include records with different
        # relative paths; both are build metadata, so keep the first.
        if getattr(old, "category", None) == "include" and getattr(value, "category", None) == "include":
            return
        raise RegistryError(f"conflicting declaration for {key} while reading {source}")


def _handle_macro(declaration: str) -> str | None:
    match = re.search(r"([A-Za-z_]\w*)\s*\(", declaration)
    return match.group(1) if match else None


def _typedef_base(element: ET.Element) -> str | None:
    """The inner base type of a basetype/bitmask typedef (``uint32_t``, ``VkFlags``)."""
    inner = element.find("type")
    return (inner.text or "").strip() if inner is not None and inner.text else None


def _func_pointer(element: ET.Element, doc: str | None, protect: str | None) -> FuncPointer:
    """Parse a ``category="funcpointer"`` typedef from its proto/param elements.

    Older registries spell a funcpointer as ``<proto><type>RET</type>
    <name>NAME</name></proto><param>...</param>``; newer ones use an inline
    ``typedef RET (VKAPI_PTR *<name>NAME</name>)(params)`` with the parameter
    names inline.  Both are normalized to a ``FuncPointer``; the inline form
    has no structured parameters, which the emitter does not need.
    """
    proto = element.find("proto")
    if proto is not None:
        name = proto.findtext("name")
        return_type = proto.findtext("type")
        if not name or not return_type:
            raise RegistryError(f"funcpointer prototype is incomplete: {_text(element)!r}")
        params = tuple(_param(param, frozenset()) for param in element.findall("param"))
        return FuncPointer(
            _generalize(name),
            name,
            strip_vk(return_type),
            return_type,
            params,
            doc,
            protect,
        )
    name = element.findtext("name")
    if not name:
        raise RegistryError(f"funcpointer is missing its name: {_text(element)!r}")
    declaration = _text(element)
    match = re.search(r"typedef\s+([A-Za-z_]\w*)\s*\(\s*VKAPI_PTR\s*\*", declaration)
    return_type = match.group(1) if match else "void"
    return FuncPointer(
        _generalize(name),
        name,
        strip_vk(return_type),
        return_type,
        (),
        doc,
        protect,
    )


def _define(name: str, declaration: str, requires: str | None, doc: str | None, protect: str | None) -> Define:
    if "#define" not in declaration:
        # Conditional blocks such as VK_USE_64_BIT_PTR_DEFINES carry raw
        # preprocessor text that is not a simple macro.
        return Define(name, name, declaration, requires, False, doc, protect)
    # Disabled only when the #define token itself is commented out; prose
    # comments before an active #define must not flip this flag.
    disabled = bool(re.search(r"//\s*#define", declaration))
    body = declaration.rsplit(name, 1)[1].strip() if name in declaration else declaration
    return Define(name, name, body, requires, disabled, doc, protect)


def _include_target(name: str, declaration: str) -> str | None:
    match = re.search(r'#include\s+"([^"]+)"', declaration)
    if match:
        return match.group(1)
    return name if not declaration else None


def _selected_extension(name: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in include) and not any(
        fnmatch.fnmatchcase(name, pattern) for pattern in exclude
    )


def _excluded(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _receiver_member_name(cpp_name: str, receiver: str, config: GeneratorConfig) -> str:
    """Receiver-relative member name.

    Vulkan command names repeat the object they operate on (vkQueueSubmit);
    once that parameter is bound to ``this`` the qualifier adds no
    information (Queue::submit).  CommandBuffer keeps the same reduction for
    the leading ``cmd`` dispatch token.  Explicit naming configuration always
    wins.
    """
    receiver_name = config.type_names.get(f"Vk{receiver}")
    if receiver_name is None:
        receiver_name = config.type_names.get(receiver, receiver)
    pascal = cpp_name[:1].upper() + cpp_name[1:]
    index = pascal.find(receiver_name)
    if index < 0:
        return cpp_name
    shortened = pascal[:index] + pascal[index + len(receiver_name):]
    if not shortened:
        return cpp_name
    return shortened[:1].lower() + shortened[1:]


def build_ir(
    paths: list[Path] | tuple[Path, ...],
    api: str = "vulkan",
    include_extensions: tuple[str, ...] = ("*",),
    exclude_extensions: tuple[str, ...] = (),
    config: GeneratorConfig | None = None,
) -> IrRegistry:
    """Parse registry XML files into the processed middle-layer IR."""
    if not paths:
        raise RegistryError("at least one registry path is required")
    config = config or GeneratorConfig()
    resolved = tuple(Path(path).resolve() for path in paths)
    registry = IrRegistry(sources=resolved, api=api)
    roots: list[ET.Element] = []
    tags: list[str] = []
    type_order: list[str] = []
    pending_struct_members: dict[str, list[ET.Element]] = {}
    for path in resolved:
        if not path.is_file():
            raise RegistryError(f"registry does not exist: {path}")
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise RegistryError(f"cannot parse registry {path}: {exc}") from exc
        if root.tag != "registry":
            raise RegistryError(f"{path} is not a Khronos registry XML file")
        roots.append(root)
        for platform in root.findall("./platforms/platform"):
            name, protect = platform.get("name"), platform.get("protect")
            if name and protect:
                registry.platforms[name] = protect
        tags.extend(tag.get("name", "") for tag in root.findall("./tags/tag") if tag.get("name"))
        for element in root.findall("./types/type"):
            if not _applies(element, api):
                continue
            name = element.get("name") or element.findtext("name")
            if not name:
                # Funcpointer typedefs carry their name inside <proto>.
                proto = element.find("proto")
                if proto is not None:
                    name = proto.findtext("name")
            if not name:
                continue
            category = element.get("category")
            declaration, doc = _split_comments(element)
            protect = element.get("protect")
            alias = element.get("alias")
            gname = _generalize(name)
            if gname not in type_order:
                type_order.append(gname)
            if alias:
                _add_unique(
                    registry.aliases, gname,
                    Alias(gname, name, _generalize(alias), category, doc, protect), path,
                )
                continue
            if category == "handle":
                handle = Handle(
                    name=gname,
                    c_name=name,
                    parents=tuple(_generalize(parent) for parent in _csv(element.get("parent"))),
                    dispatchable=_handle_macro(declaration) == "VK_DEFINE_HANDLE",
                    object_type_enum=element.get("objtypeenum"),
                    doc=doc,
                    protect=protect,
                )
                _add_unique(registry.handles, gname, handle, path)
            elif category in {"struct", "union"}:
                struct = Struct(
                    name=gname,
                    c_name=name,
                    category=category,
                    struct_extends=tuple(_generalize(item) for item in _csv(element.get("structextends"))),
                    returned_only=element.get("returnedonly") == "true",
                    allow_duplicate=element.get("allowduplicate") == "true",
                    doc=doc,
                    protect=protect,
                )
                # Members are parsed after every file is read, because member
                # direction depends on the complete set of handle names.
                pending_struct_members.setdefault(gname, []).extend(
                    member for member in element.findall("member") if _applies(member, api)
                )
                _add_unique(registry.structs, gname, struct, path)
            elif category == "bitmask":
                base = _typedef_base(element)
                bits = element.get("requires") or element.get("bitvalues")
                _add_unique(
                    registry.bitmasks, gname,
                    Bitmask(
                        gname,
                        name,
                        _generalize(base) if base else None,
                        _generalize(bits) if bits else None,
                        doc=doc,
                        protect=protect,
                    ),
                    path,
                )
            elif category == "basetype":
                _add_unique(
                    registry.basetypes, gname,
                    Basetype(gname, name, _typedef_base(element), doc, protect),
                    path,
                )
            elif category == "funcpointer":
                _add_unique(
                    registry.func_pointers, gname,
                    _func_pointer(element, doc, protect),
                    path,
                )
            elif category == "define":
                _add_unique(
                    registry.defines, gname,
                    _define(name, declaration, element.get("requires"), doc, protect),
                    path,
                )
            else:
                # Everything else (foreign/platform tokens like Display or
                # HWND, supplemental fixed-width integers, include records) is
                # retained as structured data so no input is lost.
                _add_unique(
                    registry.raw_types, gname,
                    RawType(gname, name, category, _include_target(name, declaration) if category == "include" else None, None, doc, protect),
                    path,
                )
        for group_element in root.findall("./enums"):
            if not _applies(group_element, api):
                continue
            group_name = group_element.get("name")
            if not group_name:
                continue
            group = registry.enums.setdefault(
                _generalize(group_name),
                Enum(
                    _generalize(group_name),
                    group_name,
                    group_element.get("type", "constants"),
                    int(group_element.get("bitwidth")) if group_element.get("bitwidth") else None,
                ),
            )
            if not group.doc and group_element.get("comment"):
                group.doc = group_element.get("comment")
            known = {value.name for value in group.values}
            for child in group_element.findall("enum"):
                if not _applies(child, api):
                    continue
                value = _enum_value(child)
                if value.name and value.name not in known:
                    group.values += (value,)
                    known.add(value.name)
                if group.kind == "constants" and value.name:
                    registry.constants.setdefault(
                        value.name,
                        Constant(value.name, value.name, value.value, value.alias_of, child.get("type"), value.doc, value.protect),
                    )
        for element in root.findall("./commands/command"):
            if not _applies(element, api):
                continue
            value = _command(element, api, registry.handle_names)
            if value is not None:
                _add_unique(registry.commands, value.name, value, path)
        for extension in root.findall("./extensions/extension"):
            supported = _csv(extension.get("supported"))
            if supported and api not in supported:
                continue
            extension_name = extension.get("name", "")
            if not _selected_extension(extension_name, include_extensions, exclude_extensions):
                continue
            protect = extension.get("protect") or registry.platforms.get(extension.get("platform", ""))
            for enum_element in extension.findall("require/enum"):
                if not _applies(enum_element, api):
                    continue
                extends = enum_element.get("extends")
                if extends:
                    clone = deepcopy(enum_element)
                    if clone.get("extnumber") is None and extension.get("number"):
                        clone.set("extnumber", extension.get("number", ""))
                    group = registry.enums.setdefault(_generalize(extends), Enum(_generalize(extends), extends, "enum"))
                    if clone.get("name") not in {value.name for value in group.values}:
                        group.values += (_enum_value(clone, protect),)
                elif enum_element.get("name") and (enum_element.get("alias") or enum_element.get("value")):
                    name = enum_element.get("name", "")
                    value = _enum_value(enum_element, protect)
                    registry.constants.setdefault(
                        name,
                        Constant(name, name, value.value, value.alias_of, enum_element.get("type"), value.doc, value.protect),
                    )
        for enum_element in root.findall("./feature/require/enum"):
            if not _applies(enum_element, api):
                continue
            extends = enum_element.get("extends")
            if extends:
                group = registry.enums.setdefault(_generalize(extends), Enum(_generalize(extends), extends, "enum"))
                if enum_element.get("name") not in {value.name for value in group.values}:
                    group.values += (_enum_value(enum_element),)
            elif enum_element.get("name") and (enum_element.get("alias") or enum_element.get("value")):
                name = enum_element.get("name", "")
                value = _enum_value(enum_element)
                registry.constants.setdefault(
                    name,
                    Constant(name, name, value.value, value.alias_of, enum_element.get("type"), value.doc, value.protect),
                )
    registry.tags = tuple(dict.fromkeys(tags))
    registry.type_order = tuple(type_order)

    # Parse struct members now that the complete handle set is known.
    handle_names = frozenset(registry.handles)
    for name, struct in registry.structs.items():
        members = pending_struct_members.get(name, [])
        struct.members = tuple(_param(member, handle_names) for member in members)
        _link_counts(list(struct.members))

    _apply_availability(roots, registry, api, include_extensions, exclude_extensions)
    _apply_activity(roots, registry, api, include_extensions, exclude_extensions)
    _finalize(registry, config)
    return registry


def _enum_value(element: ET.Element, protect: str | None = None) -> EnumValue:
    bitpos = element.get("bitpos")
    offset = element.get("offset")
    extnumber = element.get("extnumber")
    return EnumValue(
        name=element.get("name", ""),
        value=element.get("value"),
        bitpos=int(bitpos) if bitpos is not None else None,
        offset=int(offset) if offset is not None else None,
        extnumber=int(extnumber) if extnumber is not None else None,
        negative=element.get("dir") == "-",
        alias_of=element.get("alias"),
        doc=element.get("comment"),
        protect=element.get("protect") or protect,
    )


def _param_reparsed(placeholder: object, candidates: list[Param], handle_names: frozenset[str]) -> Param:
    del placeholder, candidates, handle_names  # legacy helper; members are re-read directly
    raise AssertionError("unused")


def _command(element: ET.Element, api: str, handle_names: frozenset[str]) -> Command | None:
    if element.get("alias"):
        name = element.get("name")
        if not name:
            return None
        return Command(_generalize(name), name, "", "", alias_of=_generalize(element.get("alias") or ""))
    proto = element.find("proto")
    if proto is None:
        return None
    name = proto.findtext("name")
    return_type = proto.findtext("type")
    if not name or not return_type:
        raise RegistryError(f"command prototype is incomplete: {_text(proto)!r}")
    declaration, doc = _split_comments(element)
    declaration = re.sub(r"\s+", " ", declaration).strip()
    command = Command(
        name=_generalize(name),
        c_name=name,
        return_type=strip_vk(return_type),
        c_return_type=return_type,
        params=tuple(_param(param, handle_names) for param in element.findall("param") if _applies(param, api)),
        export=_csv(element.get("export")),
        success_codes=_csv(element.get("successcodes")),
        error_codes=_csv(element.get("errorcodes")),
        queues=_csv(element.get("queues")),
        renderpass=element.get("renderpass"),
        command_buffer_levels=_csv(element.get("cmdbufferlevel")),
        tasks=_csv(element.get("tasks")),
        implicit_externsync=tuple(
            " ".join((param.text or "").split())
            for param in element.findall("./implicitexternsyncparams/param")
            if (param.text or "").strip()
        ),
        doc=doc,
        protect=element.get("protect"),
    )
    _link_counts(list(command.params))
    return command


# ---------------------------------------------------------------------------
# Post passes
# ---------------------------------------------------------------------------

def _apply_availability(
    roots: list[ET.Element],
    registry: IrRegistry,
    api: str,
    include_extensions: tuple[str, ...],
    exclude_extensions: tuple[str, ...],
) -> None:
    def apply(name: str, availability: Availability) -> None:
        candidates = dict.fromkeys((name, _generalize(name)))
        for collection in (
            registry.handles, registry.structs, registry.bitmasks, registry.basetypes,
            registry.func_pointers, registry.aliases, registry.commands, registry.constants, registry.enums,
        ):
            for candidate in candidates:
                item = collection.get(candidate)
                if item is not None:
                    item.availability = item.availability.merge(availability)
                    return
        for group in registry.enums.values():
            for value in group.values:
                if value.name == name:
                    value.availability = value.availability.merge(availability)
                    return

    for root in roots:
        for feature in root.findall("feature"):
            if not _applies(feature, api) or not feature.get("name"):
                continue
            for require in feature.findall("require"):
                if not _applies(require, api):
                    continue
                requirement = Availability(
                    features=(feature.get("name", ""),),
                    doc=require.get("comment") or feature.get("comment"),
                )
                for child in require:
                    name = child.get("name")
                    if name:
                        apply(name, requirement)
        for extension in root.findall("./extensions/extension"):
            supported = _csv(extension.get("supported"))
            if supported and api not in supported:
                continue
            name = extension.get("name")
            if not name or not _selected_extension(name, include_extensions, exclude_extensions):
                continue
            protect = extension.get("protect") or registry.platforms.get(extension.get("platform", ""))
            for require in extension.findall("require"):
                if not _applies(require, api):
                    continue
                requirement = Availability(
                    extensions=(name,),
                    protect=require.get("protect") or protect,
                    doc=require.get("comment") or extension.get("comment"),
                )
                for child in require:
                    child_name = child.get("name")
                    if child_name:
                        apply(child_name, requirement)


def _apply_activity(
    roots: list[ET.Element],
    registry: IrRegistry,
    api: str,
    include_extensions: tuple[str, ...],
    exclude_extensions: tuple[str, ...],
) -> None:
    """Mark entities no active feature/extension requires as inactive."""
    active: set[str] = set()
    inactive: set[str] = set()
    for root in roots:
        for feature in root.findall("feature"):
            feature_active = _applies(feature, api)
            for require in feature.findall("require"):
                target = active if feature_active and _applies(require, api) else inactive
                target.update(child.get("name", "") for child in require if child.get("name"))
        for extension in root.findall("./extensions/extension"):
            supported = _csv(extension.get("supported"))
            extension_active = (not supported or api in supported) and _selected_extension(
                extension.get("name", ""), include_extensions, exclude_extensions
            )
            for require in extension.findall("require"):
                target = active if extension_active and _applies(require, api) else inactive
                target.update(child.get("name", "") for child in require if child.get("name"))
    remove = {_generalize(name) for name in inactive - active}
    for collection in (
        registry.handles,
        registry.structs,
        registry.bitmasks,
        registry.aliases,
        registry.commands,
        registry.constants,
        registry.basetypes,
        registry.func_pointers,
        registry.defines,
        registry.raw_types,
    ):
        for name, item in collection.items():
            if name in remove:
                item.active = False
    for group in registry.enums.values():
        for value in group.values:
            if value.name in remove:
                value.active = False


def _release_target(command: Command, handle_names: frozenset[str]) -> str | None:
    """The handle type destroyed/freed/released by a lifetime command."""
    if not command.c_name.startswith(("vkDestroy", "vkFree", "vkRelease")):
        return None
    handles = [param for param in command.params if param.type in handle_names]
    pointer_targets = [param for param in handles if param.pointer_depth]
    if pointer_targets:
        return pointer_targets[-1].type if len(pointer_targets) == 1 else None
    if command.c_name.startswith("vkDestroy"):
        return handles[-1].type if handles else None
    if command.c_name.startswith("vkFree"):
        return handles[-1].type if len(handles) >= 2 else None
    # Release commands without an object after their dispatch handle are
    # operational commands, not lifetime endpoints.
    return handles[-1].type if len(handles) >= 2 else None


def _handle_releasers(registry: IrRegistry, handle_names: frozenset[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    rank = ("destroy", "free", "release")

    def precedence(command_name: str) -> int:
        return next(
            (index for index, prefix in enumerate(rank) if command_name.startswith(prefix)),
            len(rank),
        )

    for command in registry.commands.values():
        if not command.active or command.alias_of:
            continue
        target = _release_target(command, handle_names)
        if target is None:
            continue
        previous = result.get(target)
        if previous is None or precedence(command.name) < precedence(previous):
            result[target] = command.name
    return result


def _creation_infos_for_handle(registry: IrRegistry, handle: Handle) -> tuple[str, ...]:
    candidates: list[str] = []
    for command in registry.commands.values():
        if (
            not command.active
            or command.alias_of
            or not command.c_name.startswith(("vkCreate", "vkAllocate"))
        ):
            continue
        output = [param for param in command.params if param.type == handle.name and param.direction == "output"]
        if not output:
            continue
        infos = [
            param.type
            for param in command.params
            if param.const
            and param.pointer_depth
            and ("CreateInfo" in param.type or "AllocateInfo" in param.type)
            and param.type in registry.structs
        ]
        candidates.extend(infos)
    return tuple(dict.fromkeys(candidates))


def _finalize(registry: IrRegistry, config: GeneratorConfig) -> None:
    """Compute all derived relationships once parsing has completed."""
    handle_names = registry.handle_names

    # Config-driven exclusion: excluded commands and handle types are marked
    # inactive so they neither emit wrappers nor host/receive commands.
    for command in registry.commands.values():
        if _excluded(command.c_name, config.exclude_commands):
            command.active = False
    for handle in registry.handles.values():
        if _excluded(handle.c_name, config.exclude_types):
            handle.active = False
    active_handles = frozenset(
        handle.name for handle in registry.handles.values() if handle.active
    )

    # Alias resolution.
    for alias in registry.aliases.values():
        if alias.resolved_category is None:
            alias.resolved_category = registry.type_category(alias.target)

    # Direction depends on the complete handle set (including aliases), which
    # is only fully known after parsing.  Recompute it everywhere.
    for command in registry.commands.values():
        for param in command.params:
            param.direction = _direction(param, handle_names)
    for struct in registry.structs.values():
        for member in struct.members:
            member.direction = _direction(member, handle_names)

    # Resolve command aliases by materializing their target.
    for command in registry.commands.values():
        if command.alias_of and command.alias_of in registry.commands:
            target = registry.commands[command.alias_of]
            command.return_type = target.return_type
            command.c_return_type = target.c_return_type
            command.params = deepcopy(target.params)
            command.success_codes = target.success_codes
            command.error_codes = target.error_codes
            command.queues = target.queues
            command.renderpass = target.renderpass
            command.command_buffer_levels = target.command_buffer_levels
            command.tasks = target.tasks

    releasers = _handle_releasers(registry, active_handles)
    for handle in registry.handles.values():
        if not handle.active:
            continue
        handle.create_infos = _creation_infos_for_handle(registry, handle)
        handle.releaser = releasers.get(handle.name)
        if len(handle.create_infos) == 1:
            handle.create_info = handle.create_infos[0]
        elif handle.create_infos:
            handle.create_info = f"{handle.name}CreationRecord"

    # Lifetime endpoints (Destroy/Free/Release that release a handle, including
    # their aliases) are consumed by the owning handle's control-block deleter;
    # they are never emitted as public methods.
    for command in registry.commands.values():
        if command.active and _release_target(command, active_handles) is not None:
            command.active = False

    overrides = config.receivers
    for command in registry.commands.values():
        if not command.active:
            continue
        # Dispatch handle: the first handle-typed parameter.
        command.dispatch = (
            command.params[0].type
            if command.params and command.params[0].type in active_handles
            else None
        )
        # Receivers: only the dispatch handle.  Each command lives in exactly
        # one wrapper (or Context when receiver-less); additional homes are
        # requested explicitly through the receivers configuration.
        receivers: list[str] = []
        if command.dispatch:
            receivers.append(command.dispatch)
        override = overrides.get(command.c_name)
        if override:
            receivers = [value for value in receivers if value not in override.remove]
            receivers.extend(value for value in override.add if value not in receivers)
        command.receivers = tuple(receivers)
        command.cpp_name = (
            (override.rename if override and override.rename else None)
            or config.command_names.get(command.c_name)
            or strip_vk_command(command.c_name)
        )
        # A command lives in exactly one place, so it has a single member name:
        # the receiver-relative name for handle commands, cpp_name for Context.
        member = command.cpp_name
        if receivers:
            receiver = receivers[0]
            if not ((override and override.rename) or command.c_name in config.command_names):
                member = _receiver_member_name(member, receiver, config)
            if receiver == "CommandBuffer" and member.startswith("cmd") and len(member) > 3:
                member = member[3].lower() + member[4:]
        command.member_name = member

        # Outputs and two-call enumeration shape.
        outputs = tuple(
            param for param in command.params if param.direction == "output"
        )
        command.outputs = tuple(param.name for param in outputs)
        count: Param | None = None
        vector: Param | None = None
        for output in outputs:
            for length in output.lengths:
                normalized = length.text.removeprefix("latexmath:[").removesuffix("]").removeprefix("p")
                for param in command.params:
                    if param.name == length.text or param.name.lstrip("p") == normalized:
                        count, vector = param, output
                        break
        success = command.success_codes or (("VK_SUCCESS",) if command.c_return_type == "VkResult" else ())
        command.status_alternatives = len(success) > 1
        if count is not None and count.pointer_depth == 0:
            # A by-value count sizes input/output spans; it is not the
            # two-call enumeration pattern.
            count = None
            vector = None
        command.count_param = count.name if count is not None else None
        command.vector_output = vector.name if vector is not None else None
        if count is not None:
            count_name = count.name
            if count_name.startswith("p") and len(count_name) > 1 and count_name[1].isupper():
                count_name = count_name[1:]
            command.count_name = count_name[:1].lower() + count_name[1:]
        command.owned_outputs = tuple(
            param.name
            for param in outputs
            if param.type in releasers
            and command.c_name.startswith(("vkCreate", "vkAllocate", "vkAcquire", "vkRegister"))
        )

    # VK_HEADER_VERSION is a #define, not an enum constant.
    header = registry.defines.get("VK_HEADER_VERSION")
    if header is not None and header.body.isdigit():
        registry.header_version = header.body
