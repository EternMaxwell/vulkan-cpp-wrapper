from __future__ import annotations

from copy import deepcopy
import fnmatch
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .model import Availability, Command, Constant, EnumGroup, EnumValue, Member, Registry, TypeDecl


class RegistryError(ValueError):
    pass


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())


def _applies(element: ET.Element, api: str) -> bool:
    declared = _csv(element.get("api"))
    return not declared or api in declared or f"{api}base" in declared


def _text(element: ET.Element) -> str:
    # Comments are documentation, not part of the C declaration. Including them
    # makes prose like "[_DYNAMIC]" look like an array suffix.
    clone = deepcopy(element)
    for comment in clone.findall(".//comment"):
        parent = next((candidate for candidate in clone.iter() if comment in list(candidate)), None)
        if parent is not None:
            parent.remove(comment)
    return "".join(clone.itertext()).strip()


def _name(element: ET.Element) -> str | None:
    return element.get("name") or element.findtext("name")


def _pointer_depth(declaration: str, name: str) -> int:
    prefix = declaration.rsplit(name, 1)[0] if name in declaration else declaration
    return prefix.count("*")


def _member(element: ET.Element) -> Member:
    name = element.findtext("name")
    type_name = element.findtext("type")
    if not name or not type_name:
        raise RegistryError(f"member/parameter is missing a name or type: {_text(element)!r}")
    declaration = _text(element)
    prefix = declaration.rsplit(name, 1)[0]
    return Member(
        name=name,
        type=type_name,
        declaration=declaration,
        pointer_depth=_pointer_depth(declaration, name),
        const=bool(re.search(r"\bconst\b", prefix)),
        optional=_csv(element.get("optional")),
        length=_csv(element.get("len")),
        alt_length=element.get("altlen"),
        externsync=element.get("externsync") is not None,
        externsync_expression=element.get("externsync"),
        values=element.get("values"),
        selector=element.get("selector"),
        selection=element.get("selection"),
        object_type=element.get("objecttype"),
        no_auto_validity=element.get("noautovalidity") == "true",
    )


def _type(element: ET.Element, api: str) -> TypeDecl | None:
    name = _name(element)
    if not name:
        return None
    category = element.get("category")
    # Header include records are build metadata, not API type declarations, and
    # vk.xml/video.xml intentionally repeat them with slightly different text.
    if category == "include":
        return None
    return TypeDecl(
        name=name,
        category=category,
        declaration=_text(element),
        alias=element.get("alias"),
        parent=element.get("parent"),
        object_type_enum=element.get("objtypeenum"),
        struct_extends=_csv(element.get("structextends")),
        returned_only=element.get("returnedonly") == "true",
        allow_duplicate=element.get("allowduplicate") == "true",
        protect=element.get("protect"),
        # Vulkan-Headers uses `requires` for the original 32-bit masks and
        # `bitvalues` for newer masks such as VkBufferUsageFlags2.
        requires=element.get("requires") or element.get("bitvalues"),
        members=[_member(member) for member in element.findall("member") if _applies(member, api)],
    )


def _command(element: ET.Element, api: str) -> Command | None:
    if element.get("alias"):
        name = element.get("name")
        return Command(name or "", "", "", alias=element.get("alias")) if name else None
    proto = element.find("proto")
    if proto is None:
        return None
    name = proto.findtext("name")
    return_type = proto.findtext("type")
    if not name or not return_type:
        raise RegistryError(f"command prototype is incomplete: {_text(proto)!r}")
    return Command(
        name=name,
        return_type=return_type,
        declaration=_text(element),
        params=[_member(param) for param in element.findall("param") if _applies(param, api)],
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
        protect=element.get("protect"),
    )


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
        alias=element.get("alias"),
        comment=element.get("comment"),
        protect=element.get("protect") or protect,
    )


def _add_unique(target: dict[str, object], key: str, value: object, source: Path) -> None:
    old = target.get(key)
    if old is None:
        target[key] = value
    elif old != value:
        # Supplemental registries commonly repeat opaque aliases. Keep the first
        # complete declaration, but replace an earlier forward/external
        # placeholder when video.xml (or another supplemental registry)
        # supplies the actual members.
        old_decl = getattr(old, "declaration", "")
        new_decl = getattr(value, "declaration", "")
        old_members = getattr(old, "members", ())
        new_members = getattr(value, "members", ())
        old_complete = bool(old_decl or old_members)
        new_complete = bool(new_decl or new_members)
        if not old_complete and new_complete:
            target[key] = value
        elif old_complete and new_complete and old_decl != new_decl:
            raise RegistryError(f"conflicting declaration for {key} while reading {source}")


def _selected_extension(name: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in include) and not any(
        fnmatch.fnmatchcase(name, pattern) for pattern in exclude
    )


def _availability(
    root: ET.Element,
    registry: Registry,
    api: str,
    include_extensions: tuple[str, ...],
    exclude_extensions: tuple[str, ...],
) -> None:
    def apply(name: str, availability: Availability) -> None:
        for collection in (registry.types, registry.commands, registry.constants, registry.enums):
            item = collection.get(name)
            if item is not None:
                item.availability = item.availability.merge(availability)
                return
        for group in registry.enums.values():
            for value in group.values:
                if value.name == name:
                    value.availability = value.availability.merge(availability)
                    return

    for feature in root.findall("feature"):
        if not _applies(feature, api):
            continue
        feature_name = feature.get("name")
        if not feature_name:
            continue
        available = Availability(features=(feature_name,))
        for require in feature.findall("require"):
            if not _applies(require, api):
                continue
            for child in require:
                name = child.get("name")
                if name:
                    apply(name, available)
    for extension in root.findall("./extensions/extension"):
        supported = _csv(extension.get("supported"))
        if supported and api not in supported:
            continue
        name = extension.get("name")
        if not name or not _selected_extension(name, include_extensions, exclude_extensions):
            continue
        protect = extension.get("protect") or registry.platforms.get(extension.get("platform", ""))
        available = Availability(extensions=(name,), protect=protect)
        for require in extension.findall("require"):
            if not _applies(require, api):
                continue
            requirement = Availability(extensions=(name,), protect=require.get("protect") or protect)
            for child in require:
                child_name = child.get("name")
                if child_name:
                    apply(child_name, requirement)


def parse_registries(
    paths: list[Path] | tuple[Path, ...],
    api: str = "vulkan",
    include_extensions: tuple[str, ...] = ("*",),
    exclude_extensions: tuple[str, ...] = (),
) -> Registry:
    if not paths:
        raise RegistryError("at least one --registry path is required")
    resolved = tuple(Path(path).resolve() for path in paths)
    registry = Registry(sources=resolved)
    roots: list[ET.Element] = []
    tags: list[str] = []
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
            value = _type(element, api)
            if value:
                _add_unique(registry.types, value.name, value, path)
        for group_element in root.findall("./enums"):
            if not _applies(group_element, api):
                continue
            group_name = group_element.get("name")
            if not group_name:
                continue
            group = registry.enums.setdefault(
                group_name,
                EnumGroup(group_name, group_element.get("type", "constants"),
                          int(group_element.get("bitwidth")) if group_element.get("bitwidth") else None),
            )
            known = {value.name for value in group.values}
            for child in group_element.findall("enum"):
                if not _applies(child, api):
                    continue
                value = _enum_value(child)
                if value.name and value.name not in known:
                    group.values.append(value)
                    known.add(value.name)
                if group.kind == "constants" and value.name:
                    registry.constants.setdefault(value.name, Constant(value.name, value.value, value.alias, child.get("type")))
        for element in root.findall("./commands/command"):
            if not _applies(element, api):
                continue
            value = _command(element, api)
            if value:
                _add_unique(registry.commands, value.name, value, path)
        # Extension enum values may extend a named group. The numeric value for
        # offset-based entries depends on the containing extension number.
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
                    group = registry.enums.setdefault(extends, EnumGroup(extends, "enum"))
                    if clone.get("name") not in {value.name for value in group.values}:
                        group.values.append(_enum_value(clone, protect))
                elif enum_element.get("name") and (enum_element.get("alias") or enum_element.get("value")):
                    name = enum_element.get("name", "")
                    registry.constants.setdefault(
                        name,
                        Constant(
                            name,
                            enum_element.get("value"),
                            enum_element.get("alias"),
                            enum_element.get("type"),
                            protect,
                        ),
                    )
        for enum_element in root.findall("./feature/require/enum"):
            if not _applies(enum_element, api):
                continue
            extends = enum_element.get("extends")
            if extends:
                group = registry.enums.setdefault(extends, EnumGroup(extends, "enum"))
                if enum_element.get("name") not in {value.name for value in group.values}:
                    group.values.append(_enum_value(enum_element))
            elif enum_element.get("name") and (enum_element.get("alias") or enum_element.get("value")):
                name = enum_element.get("name", "")
                registry.constants.setdefault(
                    name,
                    Constant(
                        name,
                        enum_element.get("value"),
                        enum_element.get("alias"),
                        enum_element.get("type"),
                    ),
                )
    registry.tags = tuple(dict.fromkeys(tags))
    for root in roots:
        _availability(root, registry, api, include_extensions, exclude_extensions)
    _prune_inactive(registry, roots, api, include_extensions, exclude_extensions)
    _resolve_aliases(registry)
    return registry


def _prune_inactive(
    registry: Registry,
    roots: list[ET.Element],
    api: str,
    include_extensions: tuple[str, ...],
    exclude_extensions: tuple[str, ...],
) -> None:
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
    remove = inactive - active
    for collection in (registry.types, registry.commands, registry.constants):
        for name in remove:
            collection.pop(name, None)
    for group in registry.enums.values():
        group.values[:] = [value for value in group.values if value.name not in remove]


def _resolve_aliases(registry: Registry) -> None:
    for command in registry.commands.values():
        if command.alias and command.alias in registry.commands:
            target = registry.commands[command.alias]
            command.return_type = target.return_type
            command.declaration = target.declaration.replace(target.name, command.name, 1)
            command.params = deepcopy(target.params)
            command.success_codes = target.success_codes
            command.error_codes = target.error_codes
            command.queues = target.queues
            command.renderpass = target.renderpass
            command.command_buffer_levels = target.command_buffer_levels
            command.tasks = target.tasks
    for type_decl in registry.types.values():
        if type_decl.alias and type_decl.alias in registry.types and not type_decl.category:
            target = registry.types[type_decl.alias]
            type_decl.category = target.category
            type_decl.parent = target.parent
            type_decl.object_type_enum = target.object_type_enum
            type_decl.struct_extends = target.struct_extends
            type_decl.returned_only = target.returned_only
            type_decl.members = deepcopy(target.members)
