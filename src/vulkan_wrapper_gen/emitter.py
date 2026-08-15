from __future__ import annotations

from dataclasses import dataclass
import re

from .analysis import (
    ApiAnalysis,
    CommandAnalysis,
    HandleAnalysis,
    analyze,
    creation_info_for_handle,
    creation_infos_for_handle,
    handle_releasers,
    is_owned_handle_output,
    release_target,
)
from .config import GeneratorConfig
from .model import Command, EnumGroup, Member, Registry, TypeDecl
from .naming import constant_name, enum_name, flag_mask_name, strip_vk
from .template import Template
from .vma import VmaModel


def _guard(text: str, protect: str | None) -> str:
    return (
        f"#if defined({protect})\n{text}\n#endif // defined({protect})"
        if protect
        else text
    )


def _native_type(member: Member) -> str:
    prefix = "const " if member.const else ""
    # Array parameters in C declarations decay to pointers even though the XML
    # spelling has no `*` token (for example `float values[4]`).
    depth = member.pointer_depth + (1 if "[" in member.declaration else 0)
    return prefix + member.type + "*" * depth


def _public_param_name(member: Member) -> str:
    if member.pointer_depth and re.match(r"p+[A-Z]", member.name):
        base = re.sub(r"^p+(?=[A-Z])", "", member.name)
        return base[:1].lower() + base[1:]
    return member.name


def _method_name(item: CommandAnalysis, receiver: str, config: GeneratorConfig) -> str:
    """Return a receiver-relative method name.

    Vulkan command names commonly repeat the object they operate on, e.g.
    vkQueueSubmit and vkGetFenceStatus.  Once the corresponding parameter is
    bound to `this`, retaining that qualifier adds no information.  Explicit
    naming configuration always wins over this automatic reduction.
    """
    override = config.receivers.get(item.command.name)
    if (override and override.rename) or item.command.name in config.command_names:
        return item.cpp_name
    if (
        receiver == "VkCommandBuffer"
        and item.cpp_name.startswith("cmd")
        and len(item.cpp_name) > 3
    ):
        return item.cpp_name[3].lower() + item.cpp_name[4:]
    receiver_name = config.type_names.get(receiver, strip_vk(receiver))
    pascal = item.cpp_name[:1].upper() + item.cpp_name[1:]
    index = pascal.find(receiver_name)
    if index < 0:
        return item.cpp_name
    shortened = pascal[:index] + pascal[index + len(receiver_name) :]
    if not shortened:
        return item.cpp_name
    return shortened[:1].lower() + shortened[1:]


def _callable_name(
    item: CommandAnalysis, receiver: str | None, config: GeneratorConfig
) -> str:
    return (
        _method_name(item, receiver, config) if receiver is not None else item.cpp_name
    )


def _command_result_name(item: CommandAnalysis) -> str:
    return item.cpp_name[:1].upper() + item.cpp_name[1:] + "Result"


def _public_param_type(
    member: Member, registry: Registry, config: GeneratorConfig
) -> str:
    cpp = _cpp_type(member.type, registry, config)
    array_sizes = re.findall(r"\[([^\]]+)\]", member.declaration)
    if array_sizes:
        extent = ", ".join(array_sizes)
        return f"std::span<{'const ' if member.const else ''}{cpp}, {extent}>"
    if member.pointer_depth == 0 and "[" not in member.declaration:
        if (
            member.type in registry.types
            and registry.types[member.type].category == "handle"
        ):
            if "true" in member.optional:
                return f"const std::optional<{cpp}>&"
            return f"const {cpp}&"
        return cpp
    if (
        member.const
        and member.type == "char"
        and member.pointer_depth == 1
        and "null-terminated" in member.length
    ):
        return (
            "std::optional<std::string_view>"
            if "true" in member.optional
            else "std::string_view"
        )
    if member.length and "null-terminated" not in member.length:
        if member.type == "void":
            return (
                "std::span<const std::byte>" if member.const else "std::span<std::byte>"
            )
        return f"std::span<{'const ' if member.const else ''}{cpp}>"
    is_output = not member.const
    if is_output:
        if member.type == "void":
            return "void" + "*" * member.pointer_depth
        return f"{cpp}*"
    if (
        member.pointer_depth == 1
        and member.type in registry.types
        and registry.types[member.type].category in {"struct", "union"}
    ):
        if "true" in member.optional:
            return f"std::optional<std::reference_wrapper<const {cpp}>>"
        return f"const {cpp}&"
    if member.type in registry.types:
        prefix = "const " if member.const else ""
        return prefix + cpp + "*" * member.pointer_depth
    return _native_type(member)


def _public_argument(
    member: Member, registry: Registry, name: str | None = None
) -> str:
    name = name or _public_param_name(member)
    if (
        member.length and "null-terminated" not in member.length
    ) or "[" in member.declaration:
        if member.type == "void":
            return f"reinterpret_cast<{'const ' if member.const else ''}void*>({name}.empty() ? nullptr : {name}.data())"
        return f"{name}.empty() ? nullptr : {name}.data()"
    if (
        member.const
        and member.pointer_depth == 1
        and member.type in registry.types
        and registry.types[member.type].category in {"struct", "union"}
    ):
        if "true" in member.optional:
            return f"{name} ? &{name}->get() : nullptr"
        return f"&{name}"
    if member.pointer_depth == 0 and member.type in registry.types:
        category = registry.types[member.type].category
        if category == "handle":
            if "true" in member.optional:
                return f"{name} ? {name}->raw() : {member.type}{{}}"
            return f"{name}.raw()"
        if category in {"enum", "bitmask"}:
            return _native_value(member.type, name, registry)
    return name


def _needs_native_conversion(member: Member, registry: Registry) -> bool:
    if (
        member.length and "null-terminated" not in member.length
    ) or "[" in member.declaration:
        return True
    item = registry.types.get(member.type)
    if item is None:
        return False
    return item.category in {"struct", "union"} or (
        (bool(member.length) or member.pointer_depth > 0 or "[" in member.declaration)
        and item.category in {"enum", "bitmask", "handle"}
    )


def _cpp_type(type_name: str, registry: Registry, config: GeneratorConfig) -> str:
    if type_name == "VkResult":
        return "ResultCode"
    if type_name in {"VkFlags", "VkFlags64"}:
        return type_name
    if type_name in config.type_names:
        return config.type_names[type_name]
    if type_name in registry.types and registry.types[type_name].alias:
        return _cpp_type(registry.types[type_name].alias or type_name, registry, config)
    # Some extension registries provide an enum group and use it from structs
    # without retaining a corresponding <type category="enum"> declaration
    # after API/extension filtering.  The group is still an owned wrapper type.
    if type_name in registry.enums and registry.enums[type_name].kind in {
        "enum",
        "bitmask",
    }:
        return strip_vk(type_name)
    if not type_name.startswith("Vk"):
        return type_name
    if type_name in registry.types and registry.types[type_name].category in {
        "handle",
        "struct",
        "union",
        "enum",
        "bitmask",
        "basetype",
        "funcpointer",
    }:
        return strip_vk(type_name)
    return type_name


def _enum_value(value, group: EnumGroup) -> str:
    if value.alias:
        return value.alias
    if value.value is not None:
        return value.value
    if value.bitpos is not None:
        suffix = "ULL" if (group.bitwidth or 32) > 32 else "U"
        return f"(1{suffix} << {value.bitpos})"
    if value.offset is not None and value.extnumber is not None:
        number = 1_000_000_000 + (value.extnumber - 1) * 1_000 + value.offset
        return str(-number if value.negative else number)
    return value.name


def _emit_enums(registry: Registry, config: GeneratorConfig) -> str:
    output: list[str] = []
    for group in registry.enums.values():
        if group.kind not in {"enum", "bitmask"} or not group.values:
            continue
        if group.name == "VkResult" or not group.name.startswith("Vk"):
            continue
        cpp = _cpp_type(group.name, registry, config)
        underlying = "std::uint64_t" if group.bitwidth == 64 else "std::int32_t"
        values = []
        used: set[str] = set()
        for value in group.values:
            name = enum_name(group.name, value.name, registry.tags)
            if name in used:
                name += "_" + value.name.rsplit("_", 1)[-1]
            used.add(name)
            values.append(
                _guard(
                    f"    {name} = static_cast<{underlying}>({_enum_value(value, group)}),",
                    value.protect,
                )
            )
        output.append(
            _guard(
                f"enum class {cpp} : {underlying} {{\n" + "\n".join(values) + "\n};",
                group.availability.protect,
            )
        )
    return "\n\n".join(output)


def _emit_aliases(registry: Registry, config: GeneratorConfig) -> str:
    result: list[str] = []
    for item in registry.types.values():
        if item.alias:
            if item.alias not in registry.types:
                continue
            alias_cpp = config.type_names.get(item.name, strip_vk(item.name))
            target_group = registry.enums.get(item.alias)
            # Vulkan deliberately declares some reserved FlagBits names in XML
            # without a corresponding C/C++ type until the first bit exists.
            if target_group is not None and not target_group.values:
                continue
            target_cpp = _cpp_type(item.alias, registry, config)
            if alias_cpp != target_cpp:
                result.append(
                    _guard(
                        f"using {alias_cpp} = {target_cpp};",
                        item.protect or item.availability.protect,
                    )
                )
        elif item.category in {
            "basetype",
            "funcpointer",
            "bitmask",
        } and item.name.startswith("Vk"):
            if item.name in {"VkFlags", "VkFlags64"}:
                continue
            if item.category == "bitmask" and _is_typed_bitmask(item.name, registry):
                result.append(
                    _guard(
                        f"using {strip_vk(item.name)} = Flags<{_cpp_type(item.requires, registry, config)}, {item.name}>;",
                        item.protect or item.availability.protect,
                    )
                )
            else:
                result.append(
                    _guard(
                        f"using {strip_vk(item.name)} = {item.name};",
                        item.protect or item.availability.protect,
                    )
                )
        elif item.category == "union" and _has_native_definition(item):
            # Vulkan unions already provide the value semantics the wrapper
            # needs.  Alias them before owned structures are defined so fields
            # such as VkClearValue are complete at their point of use.
            result.append(
                _guard(
                    f"using {_cpp_type(item.name, registry, config)} = {item.name};",
                    item.protect or item.availability.protect,
                )
            )
    return "\n".join(result)


def _emit_constants(registry: Registry) -> str:
    lines: list[str] = []
    emitted: set[str] = set()
    for item in registry.constants.values():
        name = constant_name(item.name, registry.tags)
        if name in emitted:
            continue
        emitted.add(name)
        if item.alias:
            target = constant_name(item.alias, registry.tags)
            declaration = f"inline constexpr auto {name} = {target};"
        elif item.value is not None:
            native_type = item.type or "auto"
            declaration = (
                f"inline constexpr auto {name} = {item.name};"
                if native_type == "auto"
                else f"inline constexpr {native_type} {name} = static_cast<{native_type}>({item.name});"
            )
        else:
            continue
        lines.append(_guard(declaration, item.protect or item.availability.protect))
    return "\n".join(lines)


def _member_cpp(member: Member, registry: Registry, config: GeneratorConfig) -> str:
    value = _cpp_type(member.type, registry, config)
    array_sizes = re.findall(r"\[([^\]]+)\]", member.declaration)
    if array_sizes:
        for size in reversed(array_sizes):
            value = f"std::array<{value}, {size}>"
        return value
    if member.pointer_depth:
        if (
            member.const
            and member.type == "char"
            and member.pointer_depth == 2
            and member.length
        ):
            return "std::vector<std::string>"
        if value == "void" and member.length:
            return (
                "std::vector<std::byte>"
                if member.pointer_depth == 1
                else "std::vector<const void*>"
            )
        if (
            member.const
            and member.type == "char"
            and "null-terminated" in member.length
        ):
            return "std::string"
        if member.length and any(
            length != "null-terminated" for length in member.length
        ):
            return f"std::vector<{value}>"
        if (
            member.pointer_depth == 1
            and _type_category(member.type, registry) == "struct"
        ):
            return f"std::optional<{value}>" if "true" in member.optional else value
        if (
            member.pointer_depth == 1
            and _type_category(member.type, registry) == "native_struct"
        ):
            return f"std::optional<{value}>"
        if member.pointer_depth == 1 and _type_category(member.type, registry) in {
            "enum",
            "bitmask",
        }:
            return f"std::optional<{value}>"
        if member.pointer_depth == 1 and "true" in member.optional and value != "void":
            return f"std::optional<{value}>"
        if member.type in registry.types:
            return (
                ("const " if member.const else "") + value + "*" * member.pointer_depth
            )
        return _native_type(member)
    return value


def _safe_default(member: Member, cpp_type: str) -> str:
    if member.values:
        return f"{{static_cast<{cpp_type}>({member.values})}}"
    return "{}"


def _type_category(type_name: str, registry: Registry) -> str | None:
    item = registry.types.get(type_name)
    if item:
        return item.category
    group = registry.enums.get(type_name)
    return group.kind if group and group.kind in {"enum", "bitmask"} else None


def _is_typed_bitmask(type_name: str, registry: Registry) -> bool:
    seen: set[str] = set()
    item = registry.types.get(type_name)
    while item and item.alias and item.name not in seen:
        seen.add(item.name)
        item = registry.types.get(item.alias)
    if not item or item.category != "bitmask" or not item.requires:
        return False
    group = registry.enums.get(item.requires)
    return bool(group and group.values and item.requires.startswith("Vk"))


def _native_value(type_name: str, expression: str, registry: Registry) -> str:
    if _type_category(type_name, registry) == "bitmask" and _is_typed_bitmask(
        type_name, registry
    ):
        return f"{expression}.raw()"
    return f"static_cast<{type_name}>({expression})"


def _has_native_definition(item: TypeDecl) -> bool:
    # Supplemental registries (notably video.xml) define ordinary C structs
    # whose names do not begin with Vk.  They still need the same owned wrapper
    # treatment whenever a definition, rather than a forward placeholder, is
    # present.
    return item.category in {"struct", "union"} and bool(item.members)


def _has_pnext(type_name: str, registry: Registry) -> bool:
    item = registry.types.get(type_name)
    return bool(item and any(member.name == "pNext" for member in item.members))


def _output_chain_refresh(type_name: str, expression: str, registry: Registry) -> str:
    return (
        f" {expression}.nextInChain.refresh();"
        if _has_pnext(type_name, registry)
        else ""
    )


def _native_type_name(type_name: str, registry: Registry) -> str:
    """Name a C type without resolving to a same-named wrapper.

    VkFoo wrappers are named Foo and therefore do not shadow VkFoo.  A
    supplemental type such as StdVideoFoo keeps its public name, so its native
    spelling must be explicitly qualified from inside namespace vk.
    """
    item = registry.types.get(type_name)
    if item and _has_native_definition(item) and not type_name.startswith("Vk"):
        return f"::{type_name}"
    return type_name


def _count_sources(item: TypeDecl) -> dict[str, Member]:
    result: dict[str, Member] = {}
    member_names = {member.name for member in item.members}
    for member in item.members:
        if "[" in member.declaration:
            continue
        for length in member.length:
            if re.fullmatch(r"[A-Za-z_]\w*", length):
                result.setdefault(length, member)
        # `len` is sometimes LaTeX while `altlen` carries the machine-readable
        # form.  A quotient such as codeSize / 4 is an invertible byte-count
        # relationship, so the count is derived from the owned vector.  Do not
        # collapse semantic sizing inputs such as rasterizationSamples.
        expression = (member.alt_length or "").strip()
        quotient = re.fullmatch(r"([A-Za-z_]\w*)\s*/\s*([1-9]\d*)", expression)
        if quotient and quotient.group(1) in member_names:
            result.setdefault(quotient.group(1), member)
    return result


def _context_length_name(key: str) -> str:
    parts = key.split("_")
    return "context" + "".join(part[:1].upper() + part[1:] for part in parts)


def _direct_context_lengths(item: TypeDecl) -> tuple[str, ...]:
    result: list[str] = []
    for member in item.members:
        for expression in (
            *member.length,
            *((member.alt_length,) if member.alt_length else ()),
        ):
            for match in re.finditer(r"\*_([A-Za-z_]\w*)", expression):
                if match.group(1) not in result:
                    result.append(match.group(1))
    return tuple(result)


def _context_length_source(item: TypeDecl, key: str) -> Member | None:
    suffix = "_" + key
    return next(
        (
            member
            for member in item.members
            if member.pointer_depth == 0
            and (member.name == key or member.name.endswith(suffix))
        ),
        None,
    )


def _struct_context_lengths(
    item: TypeDecl, registry: Registry, visiting: set[str] | None = None
) -> tuple[str, ...]:
    """Contextual native array counts not stored by this structure.

    Supplemental registry expressions such as `*_max_sub_layers_minus1 + 1`
    deliberately refer to a field in an enclosing structure.  Propagate that
    requirement through nested owned records until a concrete suffix-matching
    member supplies it.
    """
    visiting = set() if visiting is None else visiting
    if item.name in visiting:
        return ()
    visiting.add(item.name)
    required = list(_direct_context_lengths(item))
    for member in item.members:
        nested = registry.types.get(member.type)
        if (
            not nested
            or nested.category != "struct"
            or not _has_native_definition(nested)
        ):
            continue
        for key in _struct_context_lengths(nested, registry, visiting):
            if _context_length_source(item, key) is None and key not in required:
                required.append(key)
    visiting.remove(item.name)
    return tuple(required)


def _native_array_size(member: Member, item: TypeDecl) -> str | None:
    """Translate registry len/altlen syntax to an expression over `native`."""
    expression = member.alt_length or next(
        (length for length in member.length if length != "null-terminated"), None
    )
    if not expression or expression == "1":
        return expression
    expression = re.sub(
        r"\*_([A-Za-z_]\w*)",
        lambda match: _context_length_name(match.group(1)),
        expression,
    )
    names = {value.name for value in item.members}
    return re.sub(
        r"\b[A-Za-z_]\w*\b",
        lambda match: (
            f"native.{match.group(0)}" if match.group(0) in names else match.group(0)
        ),
        expression,
    )


def _native_count_value(count: Member, source: Member, source_field: str) -> str:
    expression = (source.alt_length or "").strip()
    quotient = re.fullmatch(rf"{re.escape(count.name)}\s*/\s*([1-9]\d*)", expression)
    if quotient:
        return f"{source_field}.size() * {quotient.group(1)}"
    return f"{source_field}.size()"


def _safe_native_count_value(
    count: Member,
    item: TypeDecl,
    field_names: dict[str, str],
) -> str:
    """Derive a count that cannot exceed any supplied parallel array.

    Registry count members sometimes govern parallel arrays and sometimes
    mutually exclusive pointer alternatives.  The smallest non-empty capacity
    handles both: alternatives can be supplied independently, while parallel
    arrays are never exposed to native code past their backing storage.
    """
    sources = _array_members(item).get(count.name, ())
    if not sources:
        expression_source = _count_sources(item).get(count.name)
        sources = (expression_source,) if expression_source else ()
    if not sources:
        return "0"
    entries = [
        (source, _native_count_value(count, source, field_names[source.name]))
        for source in sources
        if source.name in field_names
    ]
    if not entries:
        return "0"
    if len(entries) == 1:
        return entries[0][1]
    required = [
        value
        for source, value in entries
        if not (source.optional and source.optional[0] == "true")
        and not source.no_auto_validity
    ]
    conditional = [
        value
        for source, value in entries
        if (source.optional and source.optional[0] == "true") or source.no_auto_validity
    ]
    if required:
        initial, *remaining = required
        required_loop = (
            f" for (std::size_t candidate : std::initializer_list<std::size_t>{{{', '.join(remaining)}}}) "
            "if (candidate < capacity) capacity = candidate;"
            if remaining
            else ""
        )
        conditional_loop = (
            f" for (std::size_t candidate : std::initializer_list<std::size_t>{{{', '.join(conditional)}}}) "
            "if (candidate != 0 && candidate < capacity) capacity = candidate;"
            if conditional
            else ""
        )
        return f"[&] {{ std::size_t capacity = {initial};{required_loop}{conditional_loop} return capacity; }}()"
    capacities = ", ".join(conditional)
    return (
        f"[&] {{ std::size_t capacity{{}}; for (std::size_t candidate : "
        f"std::initializer_list<std::size_t>{{{capacities}}}) "
        "if (candidate != 0 && (capacity == 0 || candidate < capacity)) capacity = candidate; "
        "return capacity; }()"
    )


def _struct_from_parent_types(
    item: TypeDecl, registry: Registry, visiting: set[str] | None = None
) -> tuple[str, ...]:
    """Concrete immediate parents needed to reconstruct handles in a struct."""
    visiting = set() if visiting is None else visiting
    if item.name in visiting:
        return ()
    visiting.add(item.name)
    result: list[str] = []
    for member in item.members:
        category = _type_category(member.type, registry)
        if category == "handle":
            handle = registry.types.get(member.type)
            parent = handle.parent.split(",")[0] if handle and handle.parent else None
            if parent and parent not in result:
                result.append(parent)
        elif category == "struct":
            nested = registry.types.get(member.type)
            if nested:
                for parent in _struct_from_parent_types(nested, registry, visiting):
                    if parent not in result:
                        result.append(parent)
    visiting.remove(item.name)
    return tuple(result)


def _from_parent_name(
    type_name: str, registry: Registry, config: GeneratorConfig
) -> str:
    return "owner" + _cpp_type(type_name, registry, config)


def _struct_from_parameters(
    item: TypeDecl, registry: Registry, config: GeneratorConfig
) -> str:
    parents = "".join(
        f", const {_cpp_type(parent, registry, config)}& {_from_parent_name(parent, registry, config)}"
        for parent in _struct_from_parent_types(item, registry)
    )
    contexts = "".join(
        f", std::size_t {_context_length_name(key)}"
        for key in _struct_context_lengths(item, registry)
    )
    return parents + contexts


def _nested_from_arguments(
    type_name: str,
    registry: Registry,
    config: GeneratorConfig,
    container: TypeDecl | None = None,
) -> str:
    nested = registry.types.get(type_name)
    if not nested:
        return ""
    parents = "".join(
        f", {_from_parent_name(parent, registry, config)}"
        for parent in _struct_from_parent_types(nested, registry)
    )
    contexts: list[str] = []
    for key in _struct_context_lengths(nested, registry):
        source = _context_length_source(container, key) if container else None
        expression = f"native.{source.name}" if source else _context_length_name(key)
        contexts.append(f", static_cast<std::size_t>({expression})")
    return parents + "".join(contexts)


def _borrow_handle_lines(
    target: str,
    source: str,
    type_name: str,
    registry: Registry,
    config: GeneratorConfig,
    indent: str = "    ",
) -> list[str]:
    cpp = _cpp_type(type_name, registry, config)
    handle = registry.types[type_name]
    parent = handle.parent.split(",")[0] if handle.parent else None
    parent_arg = f", {_from_parent_name(parent, registry, config)}" if parent else ""
    return [
        f"{indent}if ({source} == {type_name}{{}}) {target}.reset();",
        f'{indent}else {{ auto wrapped = {cpp}::borrow({source}{parent_arg}); if (wrapped) {target} = std::move(*wrapped); else {{ {target}.reset(); detail::report_error(wrapped.error(), "{cpp}", detail::raw_key({source})); }} }}',
    ]


def _array_members(item: TypeDecl) -> dict[str, list[Member]]:
    result: dict[str, list[Member]] = {}
    for member in item.members:
        if "[" in member.declaration:
            continue
        for length in member.length:
            if (
                length
                and length != "null-terminated"
                and re.fullmatch(r"[A-Za-z_]\w*", length)
            ):
                result.setdefault(length, []).append(member)
    return result


def _struct_member_names(item: TypeDecl) -> dict[str, str]:
    """Map C member names to stable C++ owned-field names.

    Pointer spelling is an implementation detail of the C ABI.  A counted
    `pProfiles` member is therefore exposed as `profiles`; its paired count is
    omitted entirely.  If Vulkan supplies both pX and ppX alternatives, the
    latter receives a `Pointers` suffix rather than colliding.
    """
    omitted = set(_count_sources(item))
    result: dict[str, str] = {}
    used: set[str] = set()
    for member in item.members:
        if member.name in omitted:
            continue
        name = member.name
        if member.pointer_depth and re.match(r"p+[A-Z]", name):
            base = re.sub(r"^p+(?=[A-Z])", "", name)
            name = base[:1].lower() + base[1:]
        if name in used:
            suffix = "Pointers" if member.pointer_depth > 1 else "Values"
            name += suffix
        used.add(name)
        result[member.name] = name
    return result


def _cstruct_cache_lines(
    item: TypeDecl, registry: Registry, config: GeneratorConfig
) -> list[str]:
    lines: list[str] = []
    for member in item.members:
        category = _type_category(member.type, registry)
        cpp = _cpp_type(member.type, registry, config)
        array_sizes = re.findall(r"\[([^\]]+)\]", member.declaration)
        if member.name == "pNext":
            continue
        if array_sizes and category == "struct":
            cache_type = f"{cpp}::CStruct"
            for size in reversed(array_sizes):
                cache_type = f"std::array<{cache_type}, {size}>"
            lines.append(f"        {cache_type} {member.name}_cache{{}};")
        if (
            member.const
            and member.type == "char"
            and member.pointer_depth == 2
            and member.length
        ):
            lines.append(f"        std::vector<const char*> {member.name}_native;")
        elif not array_sizes and member.length and category == "struct":
            lines.append(f"        std::vector<{cpp}::CStruct> {member.name}_cache;")
            lines.append(
                f"        std::vector<{_native_type_name(member.type, registry)}> {member.name}_native;"
            )
            if member.pointer_depth > 1:
                lines.append(
                    f"        std::vector<const {_native_type_name(member.type, registry)}*> {member.name}_pointers;"
                )
        elif (
            not array_sizes
            and member.length
            and category in {"handle", "enum", "bitmask"}
        ):
            lines.append(
                f"        std::vector<{_native_type_name(member.type, registry)}> {member.name}_native;"
            )
        elif member.pointer_depth == 1 and category == "struct":
            if "true" in member.optional:
                lines.append(
                    f"        std::optional<{cpp}::CStruct> {member.name}_cache;"
                )
            else:
                lines.append(f"        {cpp}::CStruct {member.name}_cache{{}};")
        elif (
            member.pointer_depth == 1
            and ("true" in member.optional or category in {"enum", "bitmask"})
            and member.type != "void"
        ):
            lines.append(
                f"        std::optional<{_native_type_name(member.type, registry)}> {member.name}_native;"
            )
        elif (
            member.pointer_depth == 0
            and category == "struct"
            and "[" not in member.declaration
        ):
            lines.append(f"        {cpp}::CStruct {member.name}_cache{{}};")
    return lines


def _emit_struct(
    item: TypeDecl, registry: Registry, config: GeneratorConfig, injection: list[str]
) -> str:
    name = _cpp_type(item.name, registry, config)
    if item.alias:
        return ""
    if not _has_native_definition(item):
        return _guard(
            f"using {name} = {item.name};", item.protect or item.availability.protect
        )
    if item.category == "union":
        return _guard(
            f"using {name} = {item.name};", item.protect or item.availability.protect
        )
    field_names = _struct_member_names(item)
    lines = [
        f"struct {name} {{",
        f"    using native_type = {_native_type_name(item.name, registry)};",
        "    static constexpr bool binary_compatible = false;",
        "    struct CStruct {",
        "        native_type value{};",
    ]
    lines.extend(_cstruct_cache_lines(item, registry, config))
    lines.extend(["    };"])
    for member in item.members:
        if member.name not in field_names:
            continue
        field_name = field_names[member.name]
        if member.name == "sType" and member.values:
            lines.append(
                f"    {_cpp_type(member.type, registry, config)} {field_name}{{static_cast<{_cpp_type(member.type, registry, config)}>({member.values})}};"
            )
        elif member.name == "pNext":
            lines.append("    ExtensionChain nextInChain{};")
        else:
            member_type = _member_cpp(member, registry, config)
            lines.append(
                f"    {member_type} {field_name}{_safe_default(member, member_type)};"
            )
    for member in item.members:
        if member.name in {"sType", "pNext"} or member.name not in field_names:
            continue
        field_name = field_names[member.name]
        method = field_name[:1].upper() + field_name[1:]
        cpp = _member_cpp(member, registry, config)
        lines.append(
            f"    {name}& set{method}({cpp} value) & {{ {field_name} = std::move(value); return *this; }}"
        )
        lines.append(
            f"    {name}&& set{method}({cpp} value) && {{ {field_name} = std::move(value); return std::move(*this); }}"
        )
    if "pNext" in {member.name for member in item.members}:
        lines.append(
            f"    template <typename T> requires StructureExtends<{name}, std::remove_cvref_t<T>>::value"
        )
        lines.append(
            f"    {name}& setNextInChain(T&& value) & {{ nextInChain.set(std::forward<T>(value)); return *this; }}"
        )
        lines.append(
            f"    template <typename T> requires StructureExtends<{name}, std::remove_cvref_t<T>>::value"
        )
        lines.append(
            f"    {name}&& setNextInChain(T&& value) && {{ nextInChain.set(std::forward<T>(value)); return std::move(*this); }}"
        )
    lines.extend(line.rstrip("\r\n") for line in injection)
    lines.extend(
        [
            "    void to_cstruct(CStruct* output) const;",
            f"    void from_cstruct(const native_type& input{_struct_from_parameters(item, registry, config)});",
            "    void from_output_cstruct(const native_type& input);",
            "};",
        ]
    )
    return _guard("\n".join(lines), item.protect or item.availability.protect)


def _emit_structs(
    registry: Registry, config: GeneratorConfig, template: Template
) -> str:
    items = [
        item
        for item in registry.structs
        if not item.alias and item.category == "struct" and _has_native_definition(item)
    ]
    names = {item.name for item in items}
    emitted: set[str] = set()
    ordered: list[TypeDecl] = []
    while len(ordered) != len(items):
        progress = False
        for item in items:
            if item.name in emitted:
                continue
            dependencies = set()
            for member in item.members:
                dependency = registry.types.get(member.type)
                dependency_name = (
                    dependency.alias if dependency and dependency.alias else member.type
                )
                if dependency_name not in names or dependency_name == item.name:
                    continue
                owns_value = member.pointer_depth == 0 or (
                    member.type in names and member.pointer_depth > 0
                )
                if owns_value:
                    dependencies.add(dependency_name)
            if dependencies <= emitted:
                ordered.append(item)
                emitted.add(item.name)
                progress = True
        if not progress:
            # Vulkan structs can have pointer cycles, but owned-value cycles are
            # impossible in C. Preserve deterministic registry order if malformed
            # supplemental input nevertheless creates one.
            ordered.extend(item for item in items if item.name not in emitted)
            break
    structs = "\n\n".join(
        _emit_struct(
            item,
            registry,
            config,
            template.injections.get(_cpp_type(item.name, registry, config), []),
        )
        for item in ordered
    )
    records: list[str] = []
    for handle in registry.handles:
        if handle.alias:
            continue
        alternatives = creation_infos_for_handle(registry, handle)
        if len(alternatives) < 2:
            continue
        name = f"{_cpp_type(handle.name, registry, config)}CreationRecord"
        lines = [
            f"struct {name} {{",
            "    using Value = std::variant<",
            "        std::monostate",
        ]
        for alternative in alternatives:
            item = registry.types[alternative]
            value = f"        , {_cpp_type(alternative, registry, config)}"
            lines.append(_guard(value, item.protect or item.availability.protect))
        lines.extend(["    >;", "    Value value{};", f"    {name}() = default;"])
        for alternative in alternatives:
            item = registry.types[alternative]
            cpp = _cpp_type(alternative, registry, config)
            lines.append(
                _guard(
                    f"    {name}(const {cpp}& input) : value(input) {{}}",
                    item.protect or item.availability.protect,
                )
            )
        lines.append("};")
        records.append("\n".join(lines))
    return structs + ("\n\n" if structs and records else "") + "\n\n".join(records)


def _to_native_scalar(member: Member, expression: str, registry: Registry) -> str:
    category = _type_category(member.type, registry)
    if category == "handle":
        return f"{expression}.raw()"
    if category in {"enum", "bitmask"}:
        return _native_value(member.type, expression, registry)
    return expression


def _emit_struct_impl(
    item: TypeDecl, registry: Registry, config: GeneratorConfig
) -> str:
    if item.alias or not _has_native_definition(item):
        return ""
    name = _cpp_type(item.name, registry, config)
    counts = _count_sources(item)
    field_names = _struct_member_names(item)
    lines = [
        f"inline void {name}::to_cstruct(CStruct* output) const {{",
        "    if (!output) return;",
        "    output->value = {};",
    ]
    for member in item.members:
        category = _type_category(member.type, registry)
        target = f"output->value.{member.name}"
        field = field_names.get(member.name, member.name)
        array_sizes = re.findall(r"\[([^\]]+)\]", member.declaration)
        if member.name == "pNext":
            lines.append(
                f"    {target} = reinterpret_cast<decltype({target})>(const_cast<void*>(nextInChain.native()));"
            )
        elif member.name in counts:
            count_value = _safe_native_count_value(member, item, field_names)
            lines.append(f"    {target} = static_cast<{member.type}>({count_value});")
        elif array_sizes:
            if category == "struct":
                lines.append(
                    f"    for (std::size_t i = 0; i < {field}.size(); ++i) {{ {field}[i].to_cstruct(&output->{member.name}_cache[i]); {target}[i] = output->{member.name}_cache[i].value; }}"
                )
            elif category == "handle":
                lines.append(
                    f"    for (std::size_t i = 0; i < {field}.size(); ++i) {target}[i] = {field}[i].raw();"
                )
            elif category in {"enum", "bitmask"}:
                lines.append(
                    f"    for (std::size_t i = 0; i < {field}.size(); ++i) {target}[i] = {_native_value(member.type, f'{field}[i]', registry)};"
                )
            else:
                lines.append(
                    f"    std::memcpy({target}, {field}.data(), sizeof({target}));"
                )
        elif (
            member.const
            and member.type == "char"
            and member.pointer_depth == 1
            and "null-terminated" in member.length
        ):
            lines.append(f"    {target} = {field}.c_str();")
        elif member.length and any(
            length != "null-terminated" for length in member.length
        ):
            if member.type == "char" and member.pointer_depth == 2:
                lines.extend(
                    [
                        f"    output->{member.name}_native.resize({field}.size());",
                        f"    for (std::size_t i = 0; i < {field}.size(); ++i) output->{member.name}_native[i] = {field}[i].c_str();",
                        f"    {target} = output->{member.name}_native.empty() ? nullptr : output->{member.name}_native.data();",
                    ]
                )
            elif category == "struct":
                cpp = _cpp_type(member.type, registry, config)
                lines.extend(
                    [
                        f"    output->{member.name}_cache.resize({field}.size());",
                        f"    output->{member.name}_native.resize({field}.size());",
                        f"    for (std::size_t i = 0; i < {field}.size(); ++i) {{ {field}[i].to_cstruct(&output->{member.name}_cache[i]); output->{member.name}_native[i] = output->{member.name}_cache[i].value; }}",
                    ]
                )
                if member.pointer_depth > 1:
                    lines.extend(
                        [
                            f"    output->{member.name}_pointers.resize(output->{member.name}_native.size());",
                            f"    for (std::size_t i = 0; i < output->{member.name}_native.size(); ++i) output->{member.name}_pointers[i] = &output->{member.name}_native[i];",
                            f"    {target} = output->{member.name}_pointers.empty() ? nullptr : output->{member.name}_pointers.data();",
                        ]
                    )
                else:
                    lines.append(
                        f"    {target} = output->{member.name}_native.empty() ? nullptr : output->{member.name}_native.data();"
                    )
            elif category in {"handle", "enum", "bitmask"}:
                transform = (
                    f"{field}[i].raw()"
                    if category == "handle"
                    else _native_value(member.type, f"{field}[i]", registry)
                )
                lines.extend(
                    [
                        f"    output->{member.name}_native.resize({field}.size());",
                        f"    for (std::size_t i = 0; i < {field}.size(); ++i) output->{member.name}_native[i] = {transform};",
                        f"    {target} = output->{member.name}_native.empty() ? nullptr : output->{member.name}_native.data();",
                    ]
                )
            else:
                pointer = f"{field}.empty() ? nullptr : {field}.data()"
                lines.append(
                    f"    {target} = reinterpret_cast<decltype({target})>(const_cast<void*>(static_cast<const void*>({pointer})));"
                )
        elif member.pointer_depth == 1 and category == "struct":
            if "true" in member.optional:
                lines.extend(
                    [
                        f"    output->{member.name}_cache.reset();",
                        f"    if ({field}) {{ output->{member.name}_cache.emplace(); {field}->to_cstruct(&*output->{member.name}_cache); }}",
                        f"    {target} = output->{member.name}_cache ? &output->{member.name}_cache->value : nullptr;",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"    {field}.to_cstruct(&output->{member.name}_cache);",
                        f"    {target} = &output->{member.name}_cache.value;",
                    ]
                )
        elif (
            member.pointer_depth == 1
            and ("true" in member.optional or category in {"enum", "bitmask"})
            and member.type != "void"
        ):
            lines.extend(
                [
                    f"    output->{member.name}_native = {field} ? std::optional<{_native_type_name(member.type, registry)}>({_native_value(member.type, f'*{field}', registry)}) : std::nullopt;",
                    f"    {target} = output->{member.name}_native ? &*output->{member.name}_native : nullptr;",
                ]
            )
        elif member.pointer_depth == 0 and category == "struct":
            lines.extend(
                [
                    f"    {field}.to_cstruct(&output->{member.name}_cache);",
                    f"    {target} = output->{member.name}_cache.value;",
                ]
            )
        elif member.pointer_depth:
            lines.append(f"    {target} = {field};")
        else:
            lines.append(
                f"    {target} = {_to_native_scalar(member, field, registry)};"
            )
    lines.extend(
        [
            "}",
            f"inline void {name}::from_cstruct(const native_type& native{_struct_from_parameters(item, registry, config)}) {{",
        ]
    )
    for member in item.members:
        category = _type_category(member.type, registry)
        source = f"native.{member.name}"
        field = field_names.get(member.name, member.name)
        array_sizes = re.findall(r"\[([^\]]+)\]", member.declaration)
        if member.name == "pNext" or member.name in counts:
            continue
        if array_sizes:
            if category == "struct":
                nested_args = _nested_from_arguments(
                    member.type, registry, config, item
                )
                lines.append(
                    f"    for (std::size_t i = 0; i < {field}.size(); ++i) {field}[i].from_cstruct({source}[i]{nested_args});"
                )
            elif category == "handle":
                lines.append(f"    for (std::size_t i = 0; i < {field}.size(); ++i) {{")
                lines.extend(
                    _borrow_handle_lines(
                        f"{field}[i]",
                        f"{source}[i]",
                        member.type,
                        registry,
                        config,
                        "        ",
                    )
                )
                lines.append("    }")
            elif category in {"enum", "bitmask"}:
                lines.append(
                    f"    for (std::size_t i = 0; i < {field}.size(); ++i) {field}[i] = static_cast<{_cpp_type(member.type, registry, config)}>({source}[i]);"
                )
            else:
                lines.append(
                    f"    std::memcpy({field}.data(), {source}, sizeof({source}));"
                )
            continue
        if (
            member.const
            and member.type == "char"
            and member.pointer_depth == 1
            and "null-terminated" in member.length
        ):
            lines.append(f'    {field} = {source} ? {source} : "";')
            continue
        if member.length and any(
            length != "null-terminated" for length in member.length
        ):
            count_source = _native_array_size(member, item)
            if not count_source:
                lines.append(f"    {field}.clear();")
                continue
            lines.append(f"    {field}.clear();")
            lines.append(f"    if ({source}) {{")
            if member.type == "char" and member.pointer_depth == 2:
                lines.append(
                    f"        {field}.reserve(static_cast<std::size_t>({count_source}));"
                )
                lines.append(
                    f'        for (std::size_t i = 0; i < static_cast<std::size_t>({count_source}); ++i) {field}.emplace_back({source}[i] ? {source}[i] : "");'
                )
            elif category == "struct":
                lines.append(
                    f"        {field}.resize(static_cast<std::size_t>({count_source}));"
                )
                native_value = (
                    f"*{source}[i]" if member.pointer_depth > 1 else f"{source}[i]"
                )
                condition = f"if ({source}[i]) " if member.pointer_depth > 1 else ""
                nested_args = _nested_from_arguments(
                    member.type, registry, config, item
                )
                lines.append(
                    f"        for (std::size_t i = 0; i < {field}.size(); ++i) {condition}{field}[i].from_cstruct({native_value}{nested_args});"
                )
            elif category == "handle":
                lines.append(
                    f"        {field}.resize(static_cast<std::size_t>({count_source}));"
                )
                lines.append(
                    f"        for (std::size_t i = 0; i < {field}.size(); ++i) {{"
                )
                lines.extend(
                    _borrow_handle_lines(
                        f"{field}[i]",
                        f"{source}[i]",
                        member.type,
                        registry,
                        config,
                        "            ",
                    )
                )
                lines.append("        }")
            elif category in {"enum", "bitmask"}:
                cpp = _cpp_type(member.type, registry, config)
                lines.append(
                    f"        {field}.resize(static_cast<std::size_t>({count_source}));"
                )
                lines.append(
                    f"        for (std::size_t i = 0; i < {field}.size(); ++i) {field}[i] = static_cast<{cpp}>({source}[i]);"
                )
            elif member.type == "void" and member.pointer_depth == 1:
                lines.append(
                    f"        {field}.resize(static_cast<std::size_t>({count_source}));"
                )
                lines.append(
                    f"        std::memcpy({field}.data(), {source}, {field}.size());"
                )
            else:
                lines.append(
                    f"        {field}.assign({source}, {source} + static_cast<std::size_t>({count_source}));"
                )
            lines.append("    }")
            continue
        if member.pointer_depth == 1 and category == "struct":
            nested_args = _nested_from_arguments(member.type, registry, config, item)
            if "true" in member.optional:
                lines.append(
                    f"    if ({source}) {{ {field}.emplace(); {field}->from_cstruct(*{source}{nested_args}); }} else {field}.reset();"
                )
            else:
                lines.append(
                    f"    if ({source}) {field}.from_cstruct(*{source}{nested_args});"
                )
            continue
        if (
            member.pointer_depth == 1
            and ("true" in member.optional or category in {"enum", "bitmask"})
            and member.type != "void"
        ):
            conversion = (
                f"static_cast<{_cpp_type(member.type, registry, config)}>(*{source})"
                if category in {"enum", "bitmask"}
                else f"*{source}"
            )
            lines.append(
                f"    if ({source}) {field} = {conversion}; else {field}.reset();"
            )
            continue
        if member.pointer_depth:
            lines.append(f"    {field} = {source};")
            continue
        if category == "handle":
            lines.extend(
                _borrow_handle_lines(field, source, member.type, registry, config)
            )
            continue
        if category in {"enum", "bitmask"}:
            lines.append(
                f"    {field} = static_cast<{_cpp_type(member.type, registry, config)}>({source});"
            )
        elif category == "struct":
            lines.append(
                f"    {field}.from_cstruct({source}{_nested_from_arguments(member.type, registry, config, item)});"
            )
        else:
            lines.append(f"    {field} = {source};")
    lines.extend(["}"])

    output_lines = [
        f"inline void {name}::from_output_cstruct(const native_type& native) {{"
    ]
    if not _struct_from_parent_types(item, registry) and not _struct_context_lengths(
        item, registry
    ):
        output_lines.append("    from_cstruct(native);")
    else:
        # Output pNext records can contain input Vulkan handles (the Metal
        # export structures are the current registry examples).  Preserve
        # those owned wrappers; refresh only fields that do not need an
        # external parent/context to reconstruct safely.
        for member in item.members:
            category = _type_category(member.type, registry)
            if member.name == "pNext" or member.name in counts or category == "handle":
                continue
            source = f"native.{member.name}"
            field = field_names.get(member.name, member.name)
            array_sizes = re.findall(r"\[([^\]]+)\]", member.declaration)
            if array_sizes:
                if category in {"enum", "bitmask"}:
                    cpp = _cpp_type(member.type, registry, config)
                    output_lines.append(
                        f"    for (std::size_t i = 0; i < {field}.size(); ++i) {field}[i] = static_cast<{cpp}>({source}[i]);"
                    )
                elif category == "struct":
                    output_lines.append(
                        f"    for (std::size_t i = 0; i < {field}.size(); ++i) {field}[i].from_output_cstruct({source}[i]);"
                    )
                else:
                    output_lines.append(
                        f"    std::memcpy({field}.data(), {source}, sizeof({source}));"
                    )
            elif member.pointer_depth or member.length:
                # Dependency-bearing extension records are input-oriented
                # except for scalar foreign output pointers.  Counted/nested
                # data would require the missing conversion context and is
                # therefore deliberately retained rather than shallow-copied.
                if not member.length and member.pointer_depth == 0:
                    output_lines.append(f"    {field} = {source};")
                elif (
                    not member.length
                    and member.pointer_depth > 0
                    and member.type not in registry.types
                ):
                    output_lines.append(f"    {field} = {source};")
            elif category in {"enum", "bitmask"}:
                output_lines.append(
                    f"    {field} = static_cast<{_cpp_type(member.type, registry, config)}>({source});"
                )
            elif category == "struct":
                output_lines.append(f"    {field}.from_output_cstruct({source});")
            else:
                output_lines.append(f"    {field} = {source};")
    if "pNext" in {member.name for member in item.members}:
        output_lines.append("    nextInChain.refresh();")
    output_lines.append("}")
    lines.extend(output_lines)
    return _guard("\n".join(lines), item.protect or item.availability.protect)


def _emit_struct_implementations(registry: Registry, config: GeneratorConfig) -> str:
    return "\n\n".join(
        _emit_struct_impl(item, registry, config)
        for item in registry.structs
        if not item.alias and item.category == "struct"
    )


def _emit_result_code(registry: Registry) -> str:
    group = registry.enums.get("VkResult")
    if group is None:
        return "enum class ResultCode : std::int32_t { Success = VK_SUCCESS };"
    lines = ["enum class ResultCode : std::int32_t {"]
    used: set[str] = set()
    for value in group.values:
        name = enum_name(group.name, value.name, registry.tags)
        if name in used:
            continue
        used.add(name)
        lines.append(
            _guard(
                f"    {name} = static_cast<std::int32_t>({_enum_value(value, group)}),",
                value.protect,
            )
        )
    lines.append("};")
    return "\n".join(lines)


def _receiver_param(item: CommandAnalysis, receiver: str | None) -> Member | None:
    if receiver is None:
        return None
    return next(
        (p for p in item.command.params if p.type == receiver and p.pointer_depth == 0),
        None,
    )


def _bound_handle_arguments(
    item: CommandAnalysis, receiver: str | None, registry: Registry
) -> dict[int, str]:
    """Map receiver and retained ancestor parameters to native expressions."""
    if receiver is None:
        return {}
    expressions: dict[str, str] = {receiver: "raw()"}
    current = registry.types.get(receiver)
    chain = ""
    visited: set[str] = set()
    while current and current.parent:
        parent_type = current.parent.split(",")[0]
        if parent_type in visited:
            break
        visited.add(parent_type)
        chain += ".parent()"
        expressions.setdefault(parent_type, f"(*this){chain}.raw()")
        current = registry.types.get(parent_type)
    result: dict[int, str] = {}
    for param in item.command.params:
        if param.pointer_depth == 0 and param.type in expressions:
            result[id(param)] = expressions[param.type]
    return result


def _bound_handle_wrapper_arguments(
    item: CommandAnalysis, receiver: str, registry: Registry
) -> dict[int, str]:
    expressions: dict[str, str] = {receiver: "*this"}
    current = registry.types.get(receiver)
    expression = "this->parent()"
    visited: set[str] = set()
    while current and current.parent:
        parent_type = current.parent.split(",")[0]
        if parent_type in visited:
            break
        visited.add(parent_type)
        expressions.setdefault(parent_type, expression)
        expression += ".parent()"
        current = registry.types.get(parent_type)
    return {
        id(param): expressions[param.type]
        for param in item.command.params
        if param.pointer_depth == 0 and param.type in expressions
    }


def _externsync_lines(
    item: CommandAnalysis,
    receiver: str | None,
    registry: Registry,
    result_type: str,
) -> list[str]:
    bound = (
        _bound_handle_wrapper_arguments(item, receiver, registry) if receiver else {}
    )
    targets: list[tuple[str, bool, bool]] = []
    dynamic_targets: list[tuple[str, str]] = []
    for param in item.command.params:
        if not param.externsync:
            continue
        if _type_category(param.type, registry) != "handle":
            expression = param.externsync_expression or ""
            nested_array = re.fullmatch(
                r"(?:maybe:)?(p[A-Z]\w*)\[\]\.([A-Za-z_]\w*)", expression
            )
            nested_member = re.fullmatch(r"(p[A-Z]\w*)->([A-Za-z_]\w*)", expression)
            struct = registry.types.get(param.type)
            if nested_array and struct is not None:
                member = next(
                    (
                        value
                        for value in struct.members
                        if value.name == nested_array.group(2)
                    ),
                    None,
                )
                if (
                    member is not None
                    and _type_category(member.type, registry) == "handle"
                ):
                    field = _struct_member_names(struct).get(member.name, member.name)
                    targets.append(
                        (f"{_public_param_name(param)}|{field}", True, False)
                    )
            elif (
                nested_member
                and struct is not None
                and nested_member.group(2) == "objectHandle"
            ):
                fields = _struct_member_names(struct)
                object_type = fields.get("objectType")
                object_handle = fields.get("objectHandle")
                if object_type and object_handle:
                    public = _public_param_name(param)
                    dynamic_targets.append(
                        (
                            f"static_cast<VkObjectType>({public}.{object_type})",
                            f"{public}.{object_handle}",
                        )
                    )
            continue
        is_bound = id(param) in bound
        expression = bound.get(id(param), _public_param_name(param))
        targets.append(
            (
                expression,
                bool(param.length) and not is_bound,
                "true" in param.optional and not is_bound,
            )
        )

    # XML prose uses implicit synchronization for a small number of parent
    # objects.  The child wrapper already knows that concrete parent, so no
    # generic object registry is needed.
    parent_exclusive = (
        bool(item.command.implicit_externsync)
        and receiver is not None
        and any("commandPool" in text for text in item.command.implicit_externsync)
    )
    receiver_exclusive = (
        bool(item.command.implicit_externsync)
        and receiver == "VkDevice"
        and any("VkQueue" in text for text in item.command.implicit_externsync)
    )
    if (
        not targets
        and not dynamic_targets
        and not parent_exclusive
        and not receiver_exclusive
    ):
        return []
    if result_type.startswith("ResultValue<"):
        failure = f"return {result_type}{{lock.error(), {{}}}};"
    elif result_type.startswith("Result<"):
        failure = "return std::unexpected(lock.error());"
    elif result_type == "void":
        failure = "detail::report_error(lock.error()); return;"
    else:
        failure = "detail::report_error(lock.error()); return {};"
    lines = ["std::vector<detail::StateLockRef> externsync_states;"]
    for index, (expression, is_span, is_optional) in enumerate(targets):
        if "|" in expression:
            span, member = expression.split("|", 1)
            lines.extend(
                [
                    f"for (const auto& value : {span}) {{",
                    f"    auto lock = detail::ExternsyncAccess::collect(value.{member}, true, externsync_states);",
                    f"    if (!lock) {{ {failure} }}",
                    "}",
                ]
            )
            continue
        if is_span:
            lines.extend(
                [
                    f"for (const auto& value : {expression}) {{",
                    "    auto lock = detail::ExternsyncAccess::collect(value, true, externsync_states);",
                    f"    if (!lock) {{ {failure} }}",
                    "}",
                ]
            )
        elif is_optional:
            lines.extend(
                [
                    f"if ({expression}) {{",
                    f"    auto lock = detail::ExternsyncAccess::collect(*{expression}, true, externsync_states);",
                    f"    if (!lock) {{ {failure} }}",
                    "}",
                ]
            )
        else:
            lines.extend(
                [
                    f"auto externsync_lock_{index} = detail::ExternsyncAccess::collect({expression}, true, externsync_states);",
                    f"if (!externsync_lock_{index}) {{ auto& lock = externsync_lock_{index}; {failure} }}",
                ]
            )
    for index, (object_type, object_handle) in enumerate(dynamic_targets):
        association = (
            "this->deviceAssociation()" if receiver else "detail::DeviceAssociation{}"
        )
        lines.extend(
            [
                f"auto externsync_dynamic_{index} = detail::ExternsyncAccess::collect({association}, {object_type}, static_cast<std::uint64_t>({object_handle}), true, externsync_states);",
                f"if (!externsync_dynamic_{index}) {{ auto& lock = externsync_dynamic_{index}; {failure} }}",
            ]
        )
    if parent_exclusive:
        lines.extend(
            [
                "auto externsync_parent = detail::ExternsyncAccess::collect(this->parent(), true, externsync_states);",
                f"if (!externsync_parent) {{ auto& lock = externsync_parent; {failure} }}",
            ]
        )
    if receiver_exclusive:
        lines.extend(
            [
                "auto externsync_receiver = detail::ExternsyncAccess::collect(*this, true, externsync_states);",
                f"if (!externsync_receiver) {{ auto& lock = externsync_receiver; {failure} }}",
            ]
        )
    lines.append("detail::StateLocks externsync_locks(externsync_states);")
    return lines


def _is_device_scope(handle: TypeDecl, registry: Registry) -> bool:
    current: TypeDecl | None = handle
    visited: set[str] = set()
    while current is not None and current.name not in visited:
        if current.name == "VkDevice":
            return True
        visited.add(current.name)
        parent = current.parent.split(",")[0] if current.parent else None
        current = registry.types.get(parent) if parent else None
    return False


def _dispatch_call(
    command: Command, receiver: str | None, registry: Registry, arguments: list[str]
) -> str:
    if receiver is None:
        return f"::{command.name}({', '.join(arguments)})"
    if command.name == "vkGetInstanceProcAddr":
        return f"::vkGetInstanceProcAddr({', '.join(arguments)})"
    dispatch_type = command.params[0].type if command.params else None
    dispatch_decl = registry.types.get(dispatch_type) if dispatch_type else None
    instance_loaded_device_commands = {
        "vkSetDebugUtilsObjectNameEXT",
        "vkSetDebugUtilsObjectTagEXT",
        "vkQueueBeginDebugUtilsLabelEXT",
        "vkQueueEndDebugUtilsLabelEXT",
        "vkQueueInsertDebugUtilsLabelEXT",
        "vkCmdBeginDebugUtilsLabelEXT",
        "vkCmdEndDebugUtilsLabelEXT",
        "vkCmdInsertDebugUtilsLabelEXT",
    }
    table = (
        "instance"
        if command.name == "vkGetDeviceProcAddr"
        or command.name in instance_loaded_device_commands
        or dispatch_decl is None
        or not _is_device_scope(dispatch_decl, registry)
        else "device"
    )
    if table == "device":
        function = f'(this->dispatchState().device ? this->dispatchState().device->{command.name} : reinterpret_cast<PFN_{command.name}>(this->dispatchState().instance->vkGetDeviceProcAddr(this->dispatchState().native_device, "{command.name}")))'
    else:
        function = f"(this->dispatchState().instance ? this->dispatchState().instance->{command.name} : ::{command.name})"
    return f"{function}({', '.join(arguments)})"


def _output_handle_parent_expression(
    handle_type: str,
    receiver: str | None,
    command_params: list[Member],
    bound: Member | None,
    registry: Registry,
) -> str | None:
    handle = registry.types.get(handle_type)
    parent_type = handle.parent.split(",")[0] if handle and handle.parent else None
    if parent_type is None:
        return None
    if receiver == parent_type:
        return "*this"
    if receiver:
        current = registry.types.get(receiver)
        expression = "this->parent()"
        while current and current.parent:
            current_parent = current.parent.split(",")[0]
            if current_parent == parent_type:
                return expression
            expression += ".parent()"
            current = registry.types.get(current_parent)
    for param in command_params:
        if (
            param is not bound
            and param.type == parent_type
            and param.pointer_depth == 0
        ):
            name = _public_param_name(param)
            return f"*{name}" if "true" in param.optional else name
    for param in command_params:
        if param is bound:
            continue
        struct = registry.types.get(param.type)
        if not struct or struct.category != "struct":
            continue
        member = next(
            (
                candidate
                for candidate in struct.members
                if candidate.type == parent_type and candidate.pointer_depth == 0
            ),
            None,
        )
        if member is None:
            continue
        struct_name = _public_param_name(param)
        field_name = _struct_member_names(struct).get(member.name, member.name)
        if param.pointer_depth and "true" in param.optional:
            return f"{struct_name}->get().{field_name}"
        return f"{struct_name}.{field_name}"
    return None


def _wrapper_expression_for_type(
    type_name: str,
    receiver: str | None,
    command_params: list[Member],
    bound: Member | None,
    registry: Registry,
) -> str | None:
    if receiver:
        current_type = receiver
        expression = "*this"
        while True:
            if current_type == type_name:
                return expression
            current = registry.types.get(current_type)
            if current is None or not current.parent:
                break
            current_type = current.parent.split(",")[0]
            expression = (
                "this->parent()" if expression == "*this" else expression + ".parent()"
            )
    for param in command_params:
        if param is bound or param.pointer_depth != 0:
            continue
        if param.type == type_name:
            return _public_param_name(param)
    for param in command_params:
        if param is bound:
            continue
        struct = registry.types.get(param.type)
        if struct is None or struct.category != "struct":
            continue
        struct_expression = _public_param_name(param)
        if param.pointer_depth and "true" in param.optional:
            struct_expression += "->get()"
        for member in struct.members:
            if (
                member.pointer_depth != 0
                or _type_category(member.type, registry) != "handle"
            ):
                continue
            expression = f"{struct_expression}.{_struct_member_names(struct).get(member.name, member.name)}"
            current_type = member.type
            while True:
                if current_type == type_name:
                    return expression
                current = registry.types.get(current_type)
                if current is None or not current.parent:
                    break
                current_type = current.parent.split(",")[0]
                expression += ".parent()"
    return None


def _command_struct_from_arguments(
    type_name: str,
    receiver: str | None,
    item: CommandAnalysis,
    bound: Member | None,
    registry: Registry,
    config: GeneratorConfig,
) -> str:
    struct = registry.types.get(type_name)
    if not struct:
        return ""
    arguments: list[str] = []
    for parent in _struct_from_parent_types(struct, registry):
        expression = _wrapper_expression_for_type(
            parent, receiver, item.command.params, bound, registry
        )
        if expression is None:
            raise ValueError(
                f"cannot infer {parent} needed to convert {type_name} returned by {item.command.name}"
            )
        arguments.append(expression)
    return "".join(f", {argument}" for argument in arguments)


def _handle_release_lambda(
    output: Member,
    producer: Command,
    release: Command,
    receiver: str | None,
    bound: Member | None,
    registry: Registry,
    config: GeneratorConfig,
) -> str | None:
    handle_types = {handle.name for handle in registry.handles}
    target = release_target(release, handle_types)
    if target is None or target.type != output.type:
        return None
    captures: list[str] = []
    arguments: list[str] = []
    setup: list[str] = []
    used_captures: set[str] = set()
    output_handle = registry.types.get(output.type)
    immediate_parent = (
        output_handle.parent.split(",")[0]
        if output_handle and output_handle.parent
        else None
    )

    def parent_expression(type_name: str) -> str | None:
        if immediate_parent is None:
            return None
        current_type = immediate_parent
        expression = "owner"
        while True:
            if current_type == type_name:
                return expression
            current = registry.types.get(current_type)
            if current is None or not current.parent:
                return None
            current_type = current.parent.split(",")[0]
            expression += ".parent()"

    for param in release.params:
        if param is target:
            arguments.append("&value" if param.pointer_depth else "value")
            continue
        if param.type == "VkAllocationCallbacks" and param.pointer_depth == 1:
            allocator_param = next(
                (
                    candidate
                    for candidate in producer.params
                    if candidate.type == param.type
                ),
                None,
            )
            if allocator_param is None:
                arguments.append("nullptr")
                continue
            public = _public_param_name(allocator_param)
            capture = "release_allocator"
            captures.append(
                f"{capture} = {public} ? std::optional<{_cpp_type(param.type, registry, config)}>({public}->get()) : std::nullopt"
            )
            setup.extend(
                [
                    f"{_cpp_type(param.type, registry, config)}::CStruct allocator_native{{}};",
                    f"if ({capture}) {capture}->to_cstruct(&allocator_native);",
                ]
            )
            arguments.append(f"{capture} ? &allocator_native.value : nullptr")
            continue
        if param.type in handle_types:
            expression = parent_expression(param.type)
            if expression is None and immediate_parent is None:
                expression = _wrapper_expression_for_type(
                    param.type, receiver, producer.params, bound, registry
                )
            if expression is None:
                return None
            if immediate_parent is not None:
                arguments.append(f"{expression}.raw()")
            else:
                capture = "release_" + _public_param_name(param)
                if capture not in used_captures:
                    captures.append(f"{capture} = {expression}")
                    used_captures.add(capture)
                arguments.append(f"{capture}.raw()")
            continue
        if param.pointer_depth == 0 and any(
            param.name in length for length in target.length
        ):
            arguments.append(f"static_cast<{param.type}>(1)")
            continue
        # Array free counts refer to the released target even though the target
        # XML length points in the opposite direction.
        if (
            param.pointer_depth == 0
            and param.name.lower().endswith("count")
            and target.pointer_depth
        ):
            arguments.append(f"static_cast<{param.type}>(1)")
            continue
        return None
    dispatch_handle = next(
        (param for param in release.params if param.type in handle_types), None
    )
    dispatch_prefix: str | None = None
    if dispatch_handle is not None:
        table = (
            "device"
            if _is_device_scope(registry.types[dispatch_handle.type], registry)
            else "instance"
        )
        if dispatch_handle is target:
            table_type = "VolkDeviceTable" if table == "device" else "VolkInstanceTable"
            loader = (
                "volkLoadDeviceTable" if table == "device" else "volkLoadInstanceTable"
            )
            setup.extend(
                [
                    f"{table_type} release_table{{}};",
                    f"{loader}(&release_table, value);",
                ]
            )
            call = f"release_table.{release.name}({', '.join(arguments)})"
        elif dispatch_handle is not target:
            dispatch_prefix = (
                parent_expression(dispatch_handle.type)
                if immediate_parent is not None
                else "release_" + _public_param_name(dispatch_handle)
            )
        if dispatch_handle is not target and dispatch_prefix:
            call = f"({dispatch_prefix}.dispatchState().{table} ? {dispatch_prefix}.dispatchState().{table}->{release.name} : ::{release.name})({', '.join(arguments)})"
        elif dispatch_handle is not target:
            call = f"::{release.name}({', '.join(arguments)})"
    else:
        call = f"::{release.name}({', '.join(arguments)})"
    body = ["try {", *(f"    {line}" for line in setup)]
    if release.return_type == "VkResult":
        body.extend(
            [
                f"    auto status = static_cast<ResultCode>({call});",
                "    if (static_cast<std::int32_t>(status) < 0) detail::report_error(status);",
            ]
        )
    else:
        body.append(f"    {call};")
    body.extend(
        [
            "} catch (...) {",
            "    detail::report_error(ResultCode::ErrorUnknown);",
            "}",
        ]
    )
    capture_list = ", ".join(captures)
    indented = " ".join(body)
    owner_parameter = (
        f"const {_cpp_type(immediate_parent, registry, config)}& owner, "
        if immediate_parent is not None
        else ""
    )
    return f"[{capture_list}]({owner_parameter}{output.type} value) noexcept {{ {indented} }}"


def _handle_ownership_condition(
    output: Member,
    producer: Command,
    registry: Registry,
    config: GeneratorConfig,
) -> str | None:
    """Return a runtime condition for handles whose individual release is optional.

    Descriptor sets are pool-owned unless the pool was explicitly created with
    FREE_DESCRIPTOR_SET.  Treat a pool with no retained creation record as the
    conservative pool-owned case; destroying the retained pool still releases
    the native sets safely.
    """
    if output.type != "VkDescriptorSet" or producer.name != "vkAllocateDescriptorSets":
        return None
    allocate_info = next(
        (
            param
            for param in producer.params
            if param.type == "VkDescriptorSetAllocateInfo"
        ),
        None,
    )
    if allocate_info is None:
        return None
    public = _public_param_name(allocate_info)
    return (
        f"{public}.descriptorPool.createInfo() && "
        f"{public}.descriptorPool.createInfo()->flags.test("
        "DescriptorPoolCreateFlagBits::FreeDescriptorSet)"
    )


def _creation_record_expression(
    output: Member,
    command: Command,
    index: str | None,
    registry: Registry,
    config: GeneratorConfig,
) -> str | None:
    handle = registry.types.get(output.type)
    if handle is None:
        return None
    record_type = creation_info_for_handle(registry, handle)
    create_infos = creation_infos_for_handle(registry, handle)
    if record_type is None:
        return None
    source = next(
        (
            param
            for param in command.params
            if param.type in create_infos and param.const and param.pointer_depth
        ),
        None,
    )
    if source is None:
        return None
    expression = _public_param_name(source)
    if source.length and index is not None:
        expression += f"[{index}]"
    elif "true" in source.optional:
        return None
    cpp = _cpp_type(record_type, registry, config)
    return f"std::make_shared<const {cpp}>({expression})"


def _command_parts(
    item: CommandAnalysis,
    receiver: str | None,
    registry: Registry,
    config: GeneratorConfig,
    value_output: Member | tuple[Member, ...] | None = None,
) -> tuple[str, str, str]:
    bound = _receiver_param(item, receiver) if receiver else None
    bound_arguments = (
        _bound_handle_arguments(item, receiver, registry) if receiver else {}
    )
    span_lengths: dict[str, Member] = {}
    for param in item.command.params:
        if (
            param.const
            and param.length
            and "null-terminated" not in param.length
            and param.type != "void"
        ):
            for length in param.length:
                if re.fullmatch(r"[A-Za-z_]\w*", length):
                    span_lengths.setdefault(length, param)

    value_outputs = (
        (value_output,) if isinstance(value_output, Member) else value_output
    ) or ()
    value_output_ids = {id(param) for param in value_outputs}
    visible = [
        param
        for param in item.command.params
        if id(param) not in bound_arguments
        and param.name not in span_lengths
        and id(param) not in value_output_ids
    ]
    params = ", ".join(
        f"{_public_param_type(param, registry, config)} {_public_param_name(param)}"
        for param in visible
    )
    value_type = None
    status_value_result = False
    if value_outputs:
        if len(value_outputs) == 1:
            cpp = _cpp_type(value_outputs[0].type, registry, config)
            value_type = f"std::vector<{cpp}>" if value_outputs[0].length else cpp
        else:
            value_type = _command_result_name(item)
        status_value_result = (
            item.command.return_type == "VkResult" and item.output.status_value
        )
        if item.command.return_type == "VkResult":
            result = (
                f"ResultValue<{value_type}>"
                if status_value_result
                else f"Result<{value_type}>"
            )
        else:
            result = value_type
    else:
        result = (
            "void"
            if item.command.return_type == "void"
            else (
                "Result<void>"
                if item.command.return_type == "VkResult"
                else _cpp_type(item.command.return_type, registry, config)
            )
        )

    prelude: list[str] = []
    postlude: list[str] = []
    failure_cleanup: list[str] = []
    arguments: list[str] = []
    wrap_failures: list[str] = []
    releasers = handle_releasers(registry, {handle.name for handle in registry.handles})
    prelude.extend(_externsync_lines(item, receiver, registry, result))
    value_locals: dict[int, str] = {}
    if value_outputs:
        if len(value_outputs) > 1:
            prelude.append(f"{value_type} value{{}};")
        for output in value_outputs:
            cpp = _cpp_type(output.type, registry, config)
            local = (
                "value"
                if len(value_outputs) == 1
                else f"result_{_public_param_name(output)}"
            )
            value_locals[id(output)] = local
            if output.length:
                size = _output_size_expression(output, item.command, registry)
                if size is None:
                    return result, params, ""
                if len(value_outputs) == 1:
                    prelude.append(
                        f"{value_type} value(static_cast<std::size_t>({size}));"
                    )
                else:
                    prelude.append(
                        f"std::vector<{cpp}> {local}(static_cast<std::size_t>({size}));"
                    )
            elif len(value_outputs) == 1:
                prelude.append(f"{value_type} value{{}};")
            else:
                prelude.append(f"{cpp} {local}{{}};")

    def double_pointer_partition(param: Member) -> tuple[str, str] | None:
        """Find the sibling struct span whose native count partitions a flat T** span."""
        # `len="count,1"` is an array of pointers to individual T values, not
        # a jagged array.  Its public representation is still a flat span, but
        # native lowering must emit one pointer for every span element.
        if any(length == "1" for length in param.length[1:]):
            return None
        outer_lengths = {
            length for length in param.length if re.fullmatch(r"[A-Za-z_]\w*", length)
        }
        for candidate in item.command.params:
            if (
                candidate is param
                or not candidate.const
                or candidate.pointer_depth != 1
            ):
                continue
            if not outer_lengths.intersection(candidate.length):
                continue
            candidate_type = registry.types.get(candidate.type)
            if candidate_type is None or candidate_type.category != "struct":
                continue
            count_member = next(
                (
                    member
                    for member in candidate_type.members
                    if member.pointer_depth == 0 and member.name.endswith("Count")
                ),
                None,
            )
            if count_member is not None:
                return _public_param_name(candidate), count_member.name
        return None

    def safe_span_count(count_name: str) -> str:
        sources: list[tuple[Member, str]] = []
        for candidate in item.command.params:
            if count_name not in candidate.length or not candidate.pointer_depth:
                continue
            # A genuine per-element secondary length is represented by one
            # flat public span and lowered separately into pointer partitions;
            # its total element count is not the outer Vulkan count.
            if (
                candidate.pointer_depth > 1
                and double_pointer_partition(candidate) is not None
            ):
                continue
            expression = (
                value_locals.get(id(candidate), _public_param_name(candidate))
                + ".size()"
            )
            sources.append((candidate, expression))
        if not sources:
            return f"{_public_param_name(span_lengths[count_name])}.size()"
        required = [
            expression
            for candidate, expression in sources
            if not (candidate.optional and candidate.optional[0] == "true")
            and not candidate.no_auto_validity
        ]
        conditional = [
            expression
            for candidate, expression in sources
            if (candidate.optional and candidate.optional[0] == "true")
            or candidate.no_auto_validity
        ]
        if required:
            initial, *remaining = required
            required_loop = (
                f" for (std::size_t candidate : std::initializer_list<std::size_t>{{{', '.join(remaining)}}}) if (candidate < capacity) capacity = candidate;"
                if remaining
                else ""
            )
            conditional_loop = (
                f" for (std::size_t candidate : std::initializer_list<std::size_t>{{{', '.join(conditional)}}}) if (candidate != 0 && candidate < capacity) capacity = candidate;"
                if conditional
                else ""
            )
            return f"[&] {{ std::size_t capacity = {initial};{required_loop}{conditional_loop} return capacity; }}()"
        capacities = ", ".join(conditional)
        return (
            f"[&] {{ std::size_t capacity{{}}; for (std::size_t candidate : std::initializer_list<std::size_t>{{{capacities}}}) "
            "if (candidate != 0 && (capacity == 0 || candidate < capacity)) capacity = candidate; return capacity; }()"
        )

    for param in item.command.params:
        is_value_output = id(param) in value_output_ids
        public = (
            value_locals[id(param)] if is_value_output else _public_param_name(param)
        )
        category = _type_category(param.type, registry)
        if id(param) in bound_arguments:
            arguments.append(bound_arguments[id(param)])
            continue
        if param.name in span_lengths:
            arguments.append(
                f"static_cast<{param.type}>({safe_span_count(param.name)})"
            )
            continue

        if (
            param.const
            and param.type == "char"
            and param.pointer_depth == 1
            and "null-terminated" in param.length
        ):
            if "true" in param.optional:
                prelude.append(
                    f"std::optional<std::string> {public}_native = {public} ? std::optional<std::string>(std::in_place, *{public}) : std::nullopt;"
                )
                arguments.append(
                    f"{public}_native ? {public}_native->c_str() : nullptr"
                )
            else:
                prelude.append(f"std::string {public}_native({public});")
                arguments.append(f"{public}_native.c_str()")
            continue

        is_span = (
            bool(param.length) and "null-terminated" not in param.length
        ) or "[" in param.declaration
        if is_span:
            if param.type == "void":
                arguments.append(
                    f"reinterpret_cast<{'const ' if param.const else ''}void*>({public}.empty() ? nullptr : {public}.data())"
                )
            elif category == "struct":
                cpp = _cpp_type(param.type, registry, config)
                if param.pointer_depth > 1:
                    partition = double_pointer_partition(param)
                    prelude.extend(
                        [
                            f"std::vector<{cpp}::CStruct> {public}_cache({public}.size());",
                            f"std::vector<{param.type}> {public}_native({public}.size());",
                            f"for (std::size_t i = 0; i < {public}.size(); ++i) {{ {public}[i].to_cstruct(&{public}_cache[i]); {public}_native[i] = {public}_cache[i].value; }}",
                        ]
                    )
                    if partition is None:
                        prelude.extend(
                            [
                                f"std::vector<const {param.type}*> {public}_pointers({public}_native.size());",
                                f"for (std::size_t i = 0; i < {public}_native.size(); ++i) {public}_pointers[i] = &{public}_native[i];",
                            ]
                        )
                    else:
                        partition_source, partition_count = partition
                        invalid_return = (
                            f"return {result}{{ResultCode::ErrorUnknown, {{}}}};"
                            if status_value_result
                            else (
                                "return std::unexpected(ResultCode::ErrorUnknown);"
                                if value_outputs
                                and item.command.return_type == "VkResult"
                                else (
                                    "return std::unexpected(ResultCode::ErrorUnknown);"
                                    if result == "Result<void>"
                                    else "detail::report_error(ResultCode::ErrorUnknown); return;"
                                )
                            )
                        )
                        prelude.extend(
                            [
                                f"std::size_t {public}_required = 0;",
                                f"for (const auto& info : {partition_source}_native) {public}_required += info.{partition_count};",
                                f"if ({public}.size() != {public}_required) {{ {invalid_return} }}",
                                f"std::vector<const {param.type}*> {public}_pointers({partition_source}_native.size());",
                                f"std::size_t {public}_offset = 0;",
                                f"for (std::size_t i = 0; i < {partition_source}_native.size(); ++i) {{",
                                f"    const auto segment_size = static_cast<std::size_t>({partition_source}_native[i].{partition_count});",
                                f"    {public}_pointers[i] = segment_size == 0 ? nullptr : {public}_native.data() + {public}_offset;",
                                f"    {public}_offset += segment_size;",
                                "}",
                            ]
                        )
                    arguments.append(
                        f"{public}_pointers.empty() ? nullptr : {public}_pointers.data()"
                    )
                else:
                    prelude.extend(
                        [
                            f"std::vector<{cpp}::CStruct> {public}_cache({public}.size());",
                            f"std::vector<{param.type}> {public}_native({public}.size());",
                        ]
                    )
                    if param.const:
                        prelude.append(
                            f"for (std::size_t i = 0; i < {public}.size(); ++i) {{ {public}[i].to_cstruct(&{public}_cache[i]); {public}_native[i] = {public}_cache[i].value; }}"
                        )
                    else:
                        prelude.append(
                            f"for (std::size_t i = 0; i < {public}.size(); ++i) {{ {public}[i].to_cstruct(&{public}_cache[i]); {public}_native[i] = {public}_cache[i].value; }}"
                        )
                        from_args = _command_struct_from_arguments(
                            param.type, receiver, item, bound, registry, config
                        )
                        postlude.append(
                            f"for (std::size_t i = 0; i < {public}.size(); ++i) {{ {public}[i].from_cstruct({public}_native[i]{from_args});{_output_chain_refresh(param.type, f'{public}[i]', registry)} }}"
                        )
                    arguments.append(
                        f"{public}_native.empty() ? nullptr : {public}_native.data()"
                    )
            elif category == "handle":
                prelude.append(
                    f"std::vector<{param.type}> {public}_native({public}.size());"
                )
                if param.const:
                    prelude.append(
                        f"for (std::size_t i = 0; i < {public}.size(); ++i) {public}_native[i] = {public}[i].raw();"
                    )
                else:
                    cpp = _cpp_type(param.type, registry, config)
                    parent = _output_handle_parent_expression(
                        param.type, receiver, item.command.params, bound, registry
                    )
                    release = releasers.get(param.type)
                    owned = release is not None and is_owned_handle_output(
                        item.command, param, releasers
                    )
                    ownership_condition = (
                        _handle_ownership_condition(
                            param, item.command, registry, config
                        )
                        if owned
                        else None
                    )
                    destroyer = (
                        _handle_release_lambda(
                            param,
                            item.command,
                            release,
                            receiver,
                            bound,
                            registry,
                            config,
                        )
                        if owned and release is not None
                        else None
                    )
                    record = (
                        _creation_record_expression(
                            param, item.command, "i", registry, config
                        )
                        if owned
                        else None
                    )
                    if owned and destroyer is None:
                        raise ValueError(
                            f"cannot infer release provenance for {item.command.name}.{param.name}"
                        )
                    if owned:
                        cleanup_call = (
                            f"{public}_cleanup({parent}, native)"
                            if parent
                            else f"{public}_cleanup(native)"
                        )
                        cleanup_lines = [
                            f"auto {public}_cleanup = {destroyer};",
                            f"for (auto native : {public}_native) if (native != {param.type}{{}}) {cleanup_call};",
                        ]
                        if ownership_condition:
                            failure_cleanup.extend(
                                [
                                    f"if ({ownership_condition}) {{",
                                    *(f"    {line}" for line in cleanup_lines),
                                    "}",
                                ]
                            )
                        else:
                            failure_cleanup.extend(cleanup_lines)
                    borrow_wrap = f"{cpp}::borrow({public}_native[i]"
                    if parent:
                        borrow_wrap += f", {parent}"
                    borrow_wrap += ")"
                    if owned:
                        adopt_wrap = f"{cpp}::makeOwned({public}_native[i]"
                        if parent:
                            adopt_wrap += f", {parent}"
                        adopt_wrap += f", {destroyer}"
                        if creation_info_for_handle(
                            registry, registry.types[param.type]
                        ):
                            adopt_wrap += f", {record or '{}'}"
                        adopt_wrap += ")"
                        wrap = (
                            f"({ownership_condition} ? {adopt_wrap} : {borrow_wrap})"
                            if ownership_condition
                            else adopt_wrap
                        )
                    else:
                        wrap = borrow_wrap
                    failure = (
                        f"return {result}{{wrapped.error(), {{}}}};"
                        if status_value_result
                        else (
                            "return std::unexpected(wrapped.error());"
                            if value_outputs and item.command.return_type == "VkResult"
                            else (
                                "return std::unexpected(wrapped.error());"
                                if result == "Result<void>"
                                else "detail::report_error(wrapped.error()); continue;"
                            )
                        )
                    )
                    if owned:
                        cleanup_call = (
                            f"cleanup({parent}, {public}_native[remaining])"
                            if parent
                            else f"cleanup({public}_native[remaining])"
                        )
                        cleanup_remainder = f"auto cleanup = {destroyer}; for (std::size_t remaining = i + 1; remaining < {public}_native.size(); ++remaining) if ({public}_native[remaining] != {param.type}{{}}) {cleanup_call};"
                        if ownership_condition:
                            cleanup_remainder = (
                                f"if ({ownership_condition}) {{ {cleanup_remainder} }}"
                            )
                        failure = cleanup_remainder + " " + failure
                    postlude.append(
                        f"for (std::size_t i = 0; i < {public}.size(); ++i) {{ if ({public}_native[i] == {param.type}{{}}) {{ {public}[i].reset(); continue; }} auto wrapped = {wrap}; if (!wrapped) {{ {failure} }} {public}[i] = std::move(*wrapped); }}"
                    )
                arguments.append(
                    f"{public}_native.empty() ? nullptr : {public}_native.data()"
                )
            elif category in {"enum", "bitmask"}:
                cpp = _cpp_type(param.type, registry, config)
                prelude.append(
                    f"std::vector<{param.type}> {public}_native({public}.size());"
                )
                if param.const:
                    prelude.append(
                        f"for (std::size_t i = 0; i < {public}.size(); ++i) {public}_native[i] = {_native_value(param.type, f'{public}[i]', registry)};"
                    )
                else:
                    postlude.append(
                        f"for (std::size_t i = 0; i < {public}.size(); ++i) {public}[i] = static_cast<{cpp}>({public}_native[i]);"
                    )
                arguments.append(
                    f"{public}_native.empty() ? nullptr : {public}_native.data()"
                )
            else:
                if param.pointer_depth > 1:
                    partition = double_pointer_partition(param)
                    if partition is None:
                        prelude.extend(
                            [
                                f"std::vector<const {param.type}*> {public}_pointers({public}.size());",
                                f"for (std::size_t i = 0; i < {public}.size(); ++i) {public}_pointers[i] = &{public}[i];",
                            ]
                        )
                    else:
                        partition_source, partition_count = partition
                        invalid_return = (
                            "return std::unexpected(ResultCode::ErrorUnknown);"
                            if result == "Result<void>"
                            else "detail::report_error(ResultCode::ErrorUnknown); return;"
                        )
                        prelude.extend(
                            [
                                f"std::size_t {public}_required = 0;",
                                f"for (const auto& info : {partition_source}_native) {public}_required += info.{partition_count};",
                                f"if ({public}.size() != {public}_required) {{ {invalid_return} }}",
                                f"std::vector<const {param.type}*> {public}_pointers({partition_source}_native.size());",
                                f"std::size_t {public}_offset = 0;",
                                f"for (std::size_t i = 0; i < {partition_source}_native.size(); ++i) {{",
                                f"    const auto segment_size = static_cast<std::size_t>({partition_source}_native[i].{partition_count});",
                                f"    {public}_pointers[i] = segment_size == 0 ? nullptr : {public}.data() + {public}_offset;",
                                f"    {public}_offset += segment_size;",
                                "}",
                            ]
                        )
                    arguments.append(
                        f"{public}_pointers.empty() ? nullptr : {public}_pointers.data()"
                    )
                else:
                    arguments.append(f"{public}.empty() ? nullptr : {public}.data()")
            continue

        if param.pointer_depth == 0:
            if category == "handle":
                arguments.append(
                    f"{public} ? {public}->raw() : {param.type}{{}}"
                    if "true" in param.optional
                    else f"{public}.raw()"
                )
            elif category in {"enum", "bitmask"}:
                arguments.append(_native_value(param.type, public, registry))
            else:
                arguments.append(public)
            continue

        if param.pointer_depth == 1 and category == "struct":
            cpp = _cpp_type(param.type, registry, config)
            prelude.append(f"{cpp}::CStruct {public}_native{{}};")
            if param.const:
                if "true" in param.optional:
                    prelude.append(
                        f"if ({public}) {public}->get().to_cstruct(&{public}_native);"
                    )
                    arguments.append(f"{public} ? &{public}_native.value : nullptr")
                else:
                    prelude.append(f"{public}.to_cstruct(&{public}_native);")
                    arguments.append(f"&{public}_native.value")
            else:
                if is_value_output:
                    prelude.append(f"{public}.to_cstruct(&{public}_native);")
                    arguments.append(f"&{public}_native.value")
                    from_args = _command_struct_from_arguments(
                        param.type, receiver, item, bound, registry, config
                    )
                    postlude.append(
                        f"{public}.from_cstruct({public}_native.value{from_args});{_output_chain_refresh(param.type, public, registry)}"
                    )
                else:
                    prelude.append(
                        f"if ({public}) {public}->to_cstruct(&{public}_native);"
                    )
                    arguments.append(f"{public} ? &{public}_native.value : nullptr")
                    from_args = _command_struct_from_arguments(
                        param.type, receiver, item, bound, registry, config
                    )
                    refresh = (
                        f" {public}->nextInChain.refresh();"
                        if _has_pnext(param.type, registry)
                        else ""
                    )
                    postlude.append(
                        f"if ({public}) {{ {public}->from_cstruct({public}_native.value{from_args});{refresh} }}"
                    )
            continue

        if param.pointer_depth == 1 and category == "handle":
            cpp = _cpp_type(param.type, registry, config)
            if param.const:
                prelude.append(
                    f"{param.type} {public}_native = {public} ? {public}->raw() : {param.type}{{}};"
                )
                arguments.append(f"&{public}_native")
            else:
                prelude.append(f"{param.type} {public}_native{{}};")
                arguments.append(
                    f"&{public}_native"
                    if is_value_output
                    else f"{public} ? &{public}_native : nullptr"
                )
                parent = _output_handle_parent_expression(
                    param.type, receiver, item.command.params, bound, registry
                )
                release = releasers.get(param.type)
                owned = release is not None and is_owned_handle_output(
                    item.command, param, releasers
                )
                ownership_condition = (
                    _handle_ownership_condition(param, item.command, registry, config)
                    if owned
                    else None
                )
                destroyer = (
                    _handle_release_lambda(
                        param, item.command, release, receiver, bound, registry, config
                    )
                    if owned and release is not None
                    else None
                )
                record = (
                    _creation_record_expression(
                        param, item.command, None, registry, config
                    )
                    if owned
                    else None
                )
                if owned and destroyer is None:
                    raise ValueError(
                        f"cannot infer release provenance for {item.command.name}.{param.name}"
                    )
                if owned:
                    cleanup_call = (
                        f"{public}_cleanup({parent}, {public}_native)"
                        if parent
                        else f"{public}_cleanup({public}_native)"
                    )
                    cleanup_lines = [
                        f"if ({public}_native != {param.type}{{}}) {{",
                        f"    auto {public}_cleanup = {destroyer};",
                        f"    {cleanup_call};",
                        "}",
                    ]
                    if ownership_condition:
                        failure_cleanup.extend(
                            [
                                f"if ({ownership_condition}) {{",
                                *(f"    {line}" for line in cleanup_lines),
                                "}",
                            ]
                        )
                    else:
                        failure_cleanup.extend(cleanup_lines)
                borrow_wrap = f"{cpp}::borrow({public}_native"
                if parent:
                    borrow_wrap += f", {parent}"
                borrow_wrap += ")"
                if owned:
                    adopt_wrap = f"{cpp}::makeOwned({public}_native"
                    if parent:
                        adopt_wrap += f", {parent}"
                    adopt_wrap += f", {destroyer}"
                    if creation_info_for_handle(registry, registry.types[param.type]):
                        adopt_wrap += f", {record or '{}'}"
                    adopt_wrap += ")"
                    wrap = (
                        f"({ownership_condition} ? {adopt_wrap} : {borrow_wrap})"
                        if ownership_condition
                        else adopt_wrap
                    )
                else:
                    wrap = borrow_wrap
                failure = (
                    f"return {result}{{wrapped.error(), {{}}}};"
                    if status_value_result
                    else (
                        "return std::unexpected(wrapped.error());"
                        if value_outputs and item.command.return_type == "VkResult"
                        else (
                            "return std::unexpected(wrapped.error());"
                            if result == "Result<void>"
                            else "detail::report_error(wrapped.error());"
                        )
                    )
                )
                if is_value_output:
                    postlude.append(
                        f"if ({public}_native != {param.type}{{}}) {{ auto wrapped = {wrap}; if (!wrapped) {{ {failure} }} else {public} = std::move(*wrapped); }}"
                    )
                else:
                    postlude.append(
                        f"if ({public}) {{ if ({public}_native == {param.type}{{}}) {public}->reset(); else {{ auto wrapped = {wrap}; if (!wrapped) {{ {failure} }} else *{public} = std::move(*wrapped); }} }}"
                    )
            continue

        if param.pointer_depth == 1 and category in {"enum", "bitmask"}:
            cpp = _cpp_type(param.type, registry, config)
            if param.const:
                prelude.append(
                    f"{param.type} {public}_native = {public} ? {_native_value(param.type, f'*{public}', registry)} : {param.type}{{}};"
                )
                arguments.append(f"{public} ? &{public}_native : nullptr")
            else:
                prelude.append(f"{param.type} {public}_native{{}};")
                arguments.append(
                    f"&{public}_native"
                    if is_value_output
                    else f"{public} ? &{public}_native : nullptr"
                )
                postlude.append(
                    f"{public} = static_cast<{cpp}>({public}_native);"
                    if is_value_output
                    else f"if ({public}) *{public} = static_cast<{cpp}>({public}_native);"
                )
            continue

        arguments.append(
            f"&{public}"
            if is_value_output
            else _public_argument(param, registry, public)
        )

    if len(value_outputs) > 1:
        postlude.extend(
            f"value.{_public_param_name(output)} = std::move({value_locals[id(output)]});"
            for output in value_outputs
        )

    if item.command.name == "vkCreateDevice":
        # Managed devices require Vulkan private data.  Work only on the
        # native conversion cache, never on the caller's DeviceCreateInfo.
        # Merge both nodes when already present so the application's chain
        # remains ordered and contains no redundant wrapper request.
        result_values = registry.enums.get("VkResult")
        overflow_error = (
            "ResultCode::ErrorTooManyObjects"
            if result_values
            and any(
                value.name == "VK_ERROR_TOO_MANY_OBJECTS"
                for value in result_values.values
            )
            else "ResultCode::ErrorUnknown"
        )
        prelude.extend(
            [
                "VkPhysicalDevicePrivateDataFeatures private_data_support{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES};",
                "VkPhysicalDeviceFeatures2 feature_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &private_data_support};",
                "auto instance_dispatch = this->dispatchState().instance;",
                "(instance_dispatch ? instance_dispatch->vkGetPhysicalDeviceFeatures2 : ::vkGetPhysicalDeviceFeatures2)(raw(), &feature_query);",
                "if (private_data_support.privateData != VK_TRUE) return std::unexpected(ResultCode::ErrorFeatureNotPresent);",
                "auto* private_data_feature = static_cast<VkPhysicalDevicePrivateDataFeatures*>(nullptr);",
                "auto* private_data_slots = static_cast<VkDevicePrivateDataCreateInfo*>(nullptr);",
                "for (auto* node = static_cast<const VkBaseInStructure*>(createInfo_native.value.pNext); node; node = node->pNext) {",
                "    if (!private_data_feature && node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES) private_data_feature = const_cast<VkPhysicalDevicePrivateDataFeatures*>(reinterpret_cast<const VkPhysicalDevicePrivateDataFeatures*>(node));",
                "    if (!private_data_slots && node->sType == VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO) private_data_slots = const_cast<VkDevicePrivateDataCreateInfo*>(reinterpret_cast<const VkDevicePrivateDataCreateInfo*>(node));",
                "}",
                "VkPhysicalDevicePrivateDataFeatures wrapper_private_data{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES};",
                "if (private_data_feature) private_data_feature->privateData = VK_TRUE;",
                "else { wrapper_private_data.privateData = VK_TRUE; wrapper_private_data.pNext = const_cast<void*>(createInfo_native.value.pNext); createInfo_native.value.pNext = &wrapper_private_data; }",
                "VkDevicePrivateDataCreateInfo wrapper_slot_request{VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO};",
                f"if (private_data_slots) {{ if (private_data_slots->privateDataSlotRequestCount == std::numeric_limits<std::uint32_t>::max()) return std::unexpected({overflow_error}); ++private_data_slots->privateDataSlotRequestCount; }}",
                "else { wrapper_slot_request.privateDataSlotRequestCount = 1; wrapper_slot_request.pNext = createInfo_native.value.pNext; createInfo_native.value.pNext = &wrapper_slot_request; }",
            ]
        )

    call = _dispatch_call(item.command, receiver, registry, arguments)
    body = list(prelude)
    if item.command.return_type == "void":
        body.append(f"{call};")
        body.extend(postlude)
        if value_outputs:
            body.append("return std::move(value);")
    elif item.command.return_type == "VkResult":
        body.append(f"auto status = static_cast<ResultCode>({call});")
        if failure_cleanup:
            body.append("if (static_cast<std::int32_t>(status) < 0) {")
            body.extend(f"    {line}" for line in failure_cleanup)
            body.append(
                f"    return {result}{{status, {{}}}};"
                if status_value_result
                else "    return std::unexpected(status);"
            )
            body.append("}")
        else:
            body.append(
                f"if (static_cast<std::int32_t>(status) < 0) return {result}{{status, {{}}}};"
                if status_value_result
                else "if (static_cast<std::int32_t>(status) < 0) return std::unexpected(status);"
            )
        body.extend(postlude)
        body.append(
            f"return {result}{{status, std::move(value)}};"
            if status_value_result
            else "return std::move(value);" if value_outputs else "return {};"
        )
    else:
        body.append(f"auto result = static_cast<{result}>({call});")
        body.extend(postlude)
        body.append("return result;")
    return result, params, "\n".join(body)


def _method_parts(
    item: CommandAnalysis, receiver: str, registry: Registry, config: GeneratorConfig
) -> tuple[str, str, str]:
    return _command_parts(item, receiver, registry, config)


def _method_decl(
    item: CommandAnalysis, receiver: str, registry: Registry, config: GeneratorConfig
) -> str:
    result, params, _ = _method_parts(item, receiver, registry, config)
    prefix = "" if result == "void" else "[[nodiscard]] "
    return (
        f"    {prefix}{result} {_method_name(item, receiver, config)}({params}) const;"
    )


def _method_impl(
    item: CommandAnalysis, receiver: str, registry: Registry, config: GeneratorConfig
) -> str:
    result, params, body = _method_parts(item, receiver, registry, config)
    if not body:
        return ""
    receiver_name = _cpp_type(receiver, registry, config)
    return f"inline {result} {receiver_name}::{_method_name(item, receiver, config)}({params}) const {{ {body} }}"


def _convenience_parts(
    item: CommandAnalysis,
    receiver: str | None,
    registry: Registry,
    config: GeneratorConfig,
) -> tuple[str, str, str] | None:
    shape = item.output
    if (
        shape.vector is None
        or shape.count is None
        or item.command.return_type not in {"VkResult", "void"}
    ):
        return None
    is_void = item.command.return_type == "void"
    bound = _receiver_param(item, receiver)
    bound_arguments = _bound_handle_arguments(item, receiver, registry)
    omitted = {*bound_arguments, id(shape.count), id(shape.vector)}
    retained = [param for param in item.command.params if id(param) not in omitted]
    params = [
        f"{_public_param_type(param, registry, config)} {_public_param_name(param)}"
        for param in retained
    ]
    count_name = shape.count_name or "count"
    params.append(f"std::uint32_t {count_name} = 0")
    value_type = (
        "std::byte"
        if shape.vector.type == "void"
        else _cpp_type(shape.vector.type, registry, config)
    )
    native_value_type = (
        "std::byte" if shape.vector.type == "void" else shape.vector.type
    )
    vector_category = _type_category(shape.vector.type, registry)
    result_type = (
        f"std::vector<{value_type}>"
        if is_void
        else (
            f"ResultValue<std::vector<{value_type}>>"
            if shape.status_value
            else f"Result<std::vector<{value_type}>>"
        )
    )
    count_type = shape.count.type
    prelude: list[str] = []
    postlude: list[str] = []
    arguments: list[str] = []
    for param in item.command.params:
        public_name = _public_param_name(param)
        if id(param) in bound_arguments:
            arguments.append(bound_arguments[id(param)])
        elif param is shape.count:
            arguments.append("&written" if param.pointer_depth else "written")
        elif param is shape.vector:
            pointer = "native_values.empty() ? nullptr : native_values.data()"
            if param.type == "void":
                pointer = f"reinterpret_cast<void*>({pointer})"
            arguments.append(pointer)
        else:
            category = _type_category(param.type, registry)
            if param.pointer_depth == 0 and category == "handle":
                arguments.append(
                    f"{public_name} ? {public_name}->raw() : {param.type}{{}}"
                    if "true" in param.optional
                    else f"{public_name}.raw()"
                )
            elif param.pointer_depth == 0 and category in {"enum", "bitmask"}:
                arguments.append(_native_value(param.type, public_name, registry))
            elif (
                param.const
                and param.type == "char"
                and param.pointer_depth == 1
                and "null-terminated" in param.length
            ):
                if "true" in param.optional:
                    prelude.append(
                        f"std::optional<std::string> {public_name}_native = {public_name} ? std::optional<std::string>(std::in_place, *{public_name}) : std::nullopt;"
                    )
                    arguments.append(
                        f"{public_name}_native ? {public_name}_native->c_str() : nullptr"
                    )
                else:
                    prelude.append(f"std::string {public_name}_native({public_name});")
                    arguments.append(f"{public_name}_native.c_str()")
            elif (
                param.length
                and "null-terminated" not in param.length
                and param.type != "void"
            ):
                if category == "struct":
                    cpp = _cpp_type(param.type, registry, config)
                    prelude.extend(
                        [
                            f"std::vector<{cpp}::CStruct> {public_name}_cache({public_name}.size());",
                            f"std::vector<{param.type}> {public_name}_native({public_name}.size());",
                        ]
                    )
                    if param.const:
                        prelude.append(
                            f"for (std::size_t i = 0; i < {public_name}.size(); ++i) {{ {public_name}[i].to_cstruct(&{public_name}_cache[i]); {public_name}_native[i] = {public_name}_cache[i].value; }}"
                        )
                    else:
                        prelude.append(
                            f"for (std::size_t i = 0; i < {public_name}.size(); ++i) {{ {public_name}[i].to_cstruct(&{public_name}_cache[i]); {public_name}_native[i] = {public_name}_cache[i].value; }}"
                        )
                        from_args = _command_struct_from_arguments(
                            param.type, receiver, item, bound, registry, config
                        )
                        postlude.append(
                            f"for (std::size_t i = 0; i < {public_name}.size(); ++i) {{ {public_name}[i].from_cstruct({public_name}_native[i]{from_args});{_output_chain_refresh(param.type, f'{public_name}[i]', registry)} }}"
                        )
                    arguments.append(
                        f"{public_name}_native.empty() ? nullptr : {public_name}_native.data()"
                    )
                elif category == "handle":
                    prelude.extend(
                        [
                            f"std::vector<{param.type}> {public_name}_native({public_name}.size());",
                            f"for (std::size_t i = 0; i < {public_name}.size(); ++i) {public_name}_native[i] = {public_name}[i].raw();",
                        ]
                    )
                    arguments.append(
                        f"{public_name}_native.empty() ? nullptr : {public_name}_native.data()"
                    )
                elif category in {"enum", "bitmask"}:
                    prelude.extend(
                        [
                            f"std::vector<{param.type}> {public_name}_native({public_name}.size());",
                            f"for (std::size_t i = 0; i < {public_name}.size(); ++i) {public_name}_native[i] = {_native_value(param.type, f'{public_name}[i]', registry)};",
                        ]
                    )
                    arguments.append(
                        f"{public_name}_native.empty() ? nullptr : {public_name}_native.data()"
                    )
                else:
                    arguments.append(
                        f"{public_name}.empty() ? nullptr : {public_name}.data()"
                    )
            elif param.pointer_depth == 1 and category == "struct":
                cpp = _cpp_type(param.type, registry, config)
                prelude.append(f"{cpp}::CStruct {public_name}_native{{}};")
                if param.const and "true" in param.optional:
                    prelude.append(
                        f"if ({public_name}) {public_name}->get().to_cstruct(&{public_name}_native);"
                    )
                    arguments.append(
                        f"{public_name} ? &{public_name}_native.value : nullptr"
                    )
                elif param.const:
                    prelude.append(f"{public_name}.to_cstruct(&{public_name}_native);")
                    arguments.append(f"&{public_name}_native.value")
                else:
                    prelude.append(
                        f"if ({public_name}) {public_name}->to_cstruct(&{public_name}_native);"
                    )
                    arguments.append(
                        f"{public_name} ? &{public_name}_native.value : nullptr"
                    )
                    from_args = _command_struct_from_arguments(
                        param.type, receiver, item, bound, registry, config
                    )
                    refresh = (
                        f" {public_name}->nextInChain.refresh();"
                        if _has_pnext(param.type, registry)
                        else ""
                    )
                    postlude.append(
                        f"if ({public_name}) {{ {public_name}->from_cstruct({public_name}_native.value{from_args});{refresh} }}"
                    )
            else:
                arguments.append(_public_argument(param, registry))
    call = _dispatch_call(item.command, receiver, registry, arguments)
    null_arguments = list(arguments)
    null_arguments[item.command.params.index(shape.vector)] = "nullptr"
    null_call = _dispatch_call(item.command, receiver, registry, null_arguments)
    struct_storage = []
    struct_prepare = []
    if vector_category == "struct":
        struct_storage = [
            f"        std::vector<{value_type}> values;",
            f"        std::vector<{value_type}::CStruct> values_cache;",
            "        auto prepare_native_values = [&] {",
            "            values.resize(native_values.size());",
            "            values_cache.resize(native_values.size());",
            "            for (std::size_t i = 0; i < native_values.size(); ++i) { values[i].to_cstruct(&values_cache[i]); native_values[i] = values_cache[i].value; }",
            "        };",
        ]
        struct_prepare = ["        prepare_native_values();"]
    if shape.count.pointer_depth:
        if is_void:
            body = [
                f"        {count_type} written = static_cast<{count_type}>({count_name});",
                f"        std::vector<{native_value_type}> native_values;",
                *struct_storage,
                f"        if ({count_name} == 0) {null_call};",
                "        native_values.resize(written);",
                *struct_prepare,
                "        written = static_cast<"
                + count_type
                + ">(native_values.size());",
                f"        {call};",
                "        native_values.resize(std::min<std::size_t>(native_values.size(), static_cast<std::size_t>(written)));",
            ]
        else:
            body = [
                f"        {count_type} written = static_cast<{count_type}>({count_name});",
                f"        std::vector<{native_value_type}> native_values;",
                *struct_storage,
                f"        ResultCode status{{ResultCode::Success}};",
                f"        if ({count_name} == 0) {{",
                f"            status = static_cast<ResultCode>({null_call});",
                "            if (static_cast<std::int32_t>(status) < 0) "
                + (
                    "return ResultValue<std::vector<" + value_type + ">>{status, {}};"
                    if shape.status_value
                    else "return std::unexpected(status);"
                ),
                "        }",
                "        native_values.resize(written);",
                "        do {",
                "            written = static_cast<"
                + count_type
                + ">(native_values.size());",
                *("            prepare_native_values();" for _ in struct_prepare),
                f"            status = static_cast<ResultCode>({call});",
                "            native_values.resize(std::min<std::size_t>(native_values.size(), static_cast<std::size_t>(written)));",
                "            if (status != ResultCode::Incomplete || "
                + count_name
                + " != 0) break;",
                "            " + count_type + " required{};",
            ]
            retry_args = list(null_arguments)
            retry_args[item.command.params.index(shape.count)] = "&required"
            body.extend(
                [
                    f"            status = static_cast<ResultCode>({_dispatch_call(item.command, receiver, registry, retry_args)});",
                    "            if (static_cast<std::int32_t>(status) < 0) break;",
                    "            native_values.resize(required);",
                    "        } while (true);",
                ]
            )
    else:
        body = [
            f"        {count_type} written = static_cast<{count_type}>({count_name});",
            f"        std::vector<{native_value_type}> native_values(written);",
            *struct_storage,
            *struct_prepare,
            f"        auto status = static_cast<ResultCode>({call});",
        ]
    body.extend(f"        {line}" for line in postlude)
    category = vector_category
    if shape.vector.type == "void":
        body.append("        auto values = std::move(native_values);")
    elif category == "struct":
        from_args = _command_struct_from_arguments(
            shape.vector.type, receiver, item, bound, registry, config
        )
        body.extend(
            [
                "        values.resize(native_values.size());",
                f"        for (std::size_t i = 0; i < values.size(); ++i) {{ values[i].from_cstruct(native_values[i]{from_args});{_output_chain_refresh(shape.vector.type, 'values[i]', registry)} }}",
            ]
        )
    elif category in {"enum", "bitmask"}:
        body.extend(
            [
                f"        std::vector<{value_type}> values(native_values.size());",
                f"        for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<{value_type}>(native_values[i]);",
            ]
        )
    elif category == "union":
        body.append("        auto values = std::move(native_values);")
    elif category == "handle":
        # Enumeration commands return borrowed handles.  Resolve the concrete
        # parent from the receiver where possible; no owner template is used.
        parent_expr = _output_handle_parent_expression(
            shape.vector.type, receiver, item.command.params, bound, registry
        )
        borrow_args = "native_values[i]" + (f", {parent_expr}" if parent_expr else "")
        body.extend(
            [
                f"        std::vector<{value_type}> values;",
                "        values.reserve(native_values.size());",
                f"        for (std::size_t i = 0; i < native_values.size(); ++i) {{ auto wrapped = {value_type}::borrow({borrow_args}); if (!wrapped) "
                + (
                    f"return ResultValue<std::vector<{value_type}>>{{wrapped.error(), {{}}}};"
                    if shape.status_value
                    else "return std::unexpected(wrapped.error());"
                )
                + " values.push_back(std::move(*wrapped)); }",
            ]
        )
    else:
        body.append(
            f"        std::vector<{value_type}> values(native_values.begin(), native_values.end());"
        )
    if is_void:
        body.append("        return values;")
    elif shape.status_value:
        body.append(
            f"        return ResultValue<std::vector<{value_type}>>{{status, std::move(values)}};"
        )
    else:
        body.extend(
            [
                "        if (static_cast<std::int32_t>(status) < 0) return std::unexpected(status);",
                "        return values;",
            ]
        )
    body = [*(f"        {line}" for line in prelude), *body]
    return (
        result_type,
        ", ".join(params),
        "\n".join(line[8:] if line.startswith("        ") else line for line in body),
    )


def _convenience_decl(
    item: CommandAnalysis,
    receiver: str | None,
    registry: Registry,
    config: GeneratorConfig,
) -> str | None:
    parts = _convenience_parts(item, receiver, registry, config)
    if parts is None:
        return None
    result, params, _ = parts
    return f"    [[nodiscard]] {result} {_callable_name(item, receiver, config)}({params}) const;"


def _convenience_impl(
    item: CommandAnalysis,
    receiver: str | None,
    registry: Registry,
    config: GeneratorConfig,
) -> str | None:
    parts = _convenience_parts(item, receiver, registry, config)
    if parts is None:
        return None
    result, params, body = parts
    params = re.sub(r"\s*=\s*0(?=,|$)", "", params)
    receiver_name = (
        _cpp_type(receiver, registry, config) if receiver is not None else "Context"
    )
    indented = "\n".join(f"    {line}" if line else "" for line in body.splitlines())
    return f"inline {result} {receiver_name}::{_callable_name(item, receiver, config)}({params}) const {{\n{indented}\n}}"


def _output_size_expression(
    output: Member, command: Command, registry: Registry
) -> str | None:
    if not output.length:
        return None
    length = next(
        (value for value in output.length if value != "null-terminated"), None
    )
    if length is None:
        return None
    if re.fullmatch(r"[A-Za-z_]\w*", length):
        count = next((param for param in command.params if param.name == length), None)
        if count is None:
            return None
        source = next(
            (
                param
                for param in command.params
                if param is not output
                and param.const
                and length in param.length
                and param.pointer_depth
            ),
            None,
        )
        if source is not None:
            return f"{_public_param_name(source)}.size()"
        return _public_param_name(count)
    match = re.fullmatch(r"(p[A-Z]\w*)->([A-Za-z_]\w*)", length)
    if match:
        param_name, member_name = match.groups()
        param = next(
            (candidate for candidate in command.params if candidate.name == param_name),
            None,
        )
        if param is None:
            return None
        struct = registry.types.get(param.type)
        if struct is None or struct.category != "struct":
            return None
        public = _public_param_name(param)
        field_names = _struct_member_names(struct)
        if member_name in field_names:
            return f"{public}.{field_names[member_name]}"
        source = _count_sources(struct).get(member_name)
        if source is not None:
            return f"{public}.{field_names[source.name]}.size()"
    return None


def _owned_handle_convenience_parts(
    item: CommandAnalysis,
    receiver: str | None,
    registry: Registry,
    config: GeneratorConfig,
) -> tuple[str, str, str] | None:
    if item.command.return_type not in {"VkResult", "void"}:
        return None
    # Enumeration overloads have their own status-preserving retry path.
    # Other commands with multiple success codes also need a direct native
    # convenience path so a meaningful positive status is not discarded by
    # the Result<void> raw overload.
    if item.output.vector is not None:
        return None
    if len(item.output.outputs) != 1:
        return None
    output = item.output.outputs[0]
    if output.type == "void":
        return None
    if item.command.return_type == "VkResult" and len(item.command.success_codes) > 1:
        result, params, body = _command_parts(item, receiver, registry, config, output)
        return (result, params, body) if body else None
    bound_arguments = _bound_handle_arguments(item, receiver, registry)
    span_lengths: set[str] = set()
    for param in item.command.params:
        if param.const and param.length and "null-terminated" not in param.length:
            span_lengths.update(
                length
                for length in param.length
                if re.fullmatch(r"[A-Za-z_]\w*", length)
            )
    visible = [
        param
        for param in item.command.params
        if id(param) not in bound_arguments and param.name not in span_lengths
    ]
    retained = [param for param in visible if param is not output]
    params = ", ".join(
        f"{_public_param_type(param, registry, config)} {_public_param_name(param)}"
        for param in retained
    )
    cpp = _cpp_type(output.type, registry, config)
    size = _output_size_expression(output, item.command, registry)
    method = _callable_name(item, receiver, config)
    output_argument = "values" if output.length else "&value"
    call_arguments = ", ".join(
        output_argument if param is output else _public_param_name(param)
        for param in visible
    )
    if output.length:
        if size is None:
            return None
        value_type = f"std::vector<{cpp}>"
        if item.command.return_type == "VkResult":
            result = f"Result<{value_type}>"
            body = (
                f"{value_type} values(static_cast<std::size_t>({size})); auto status = {method}({call_arguments}); "
                "if (!status) return std::unexpected(status.error()); return values;"
            )
        else:
            result = value_type
            body = f"{value_type} values(static_cast<std::size_t>({size})); {method}({call_arguments}); return values;"
    else:
        if item.command.return_type == "VkResult":
            result = f"Result<{cpp}>"
            body = (
                f"{cpp} value{{}}; auto status = {method}({call_arguments}); "
                "if (!status) return std::unexpected(status.error()); return value;"
            )
        else:
            result = cpp
            body = f"{cpp} value{{}}; {method}({call_arguments}); return value;"
    return result, params, body


def _owned_handle_convenience_decl(
    item: CommandAnalysis,
    receiver: str | None,
    registry: Registry,
    config: GeneratorConfig,
) -> str | None:
    parts = _owned_handle_convenience_parts(item, receiver, registry, config)
    if parts is None:
        return None
    result, params, _ = parts
    return f"    [[nodiscard]] {result} {_callable_name(item, receiver, config)}({params}) const;"


def _owned_handle_convenience_impl(
    item: CommandAnalysis,
    receiver: str | None,
    registry: Registry,
    config: GeneratorConfig,
) -> str | None:
    parts = _owned_handle_convenience_parts(item, receiver, registry, config)
    if parts is None:
        return None
    result, params, body = parts
    receiver_name = (
        _cpp_type(receiver, registry, config) if receiver is not None else "Context"
    )
    return f"inline {result} {receiver_name}::{_callable_name(item, receiver, config)}({params}) const {{ {body} }}"


def _multi_output_parts(
    item: CommandAnalysis,
    receiver: str | None,
    registry: Registry,
    config: GeneratorConfig,
) -> tuple[str, str, str] | None:
    outputs = item.output.outputs
    if (
        item.command.return_type != "VkResult"
        or item.output.vector is not None
        or len(outputs) < 2
    ):
        return None
    if any(
        output.type == "void"
        or (
            output.length
            and _output_size_expression(output, item.command, registry) is None
        )
        for output in outputs
    ):
        return None
    result, params, body = _command_parts(item, receiver, registry, config, outputs)
    return (result, params, body) if body else None


def _has_multi_output_result(item: CommandAnalysis, registry: Registry) -> bool:
    outputs = item.output.outputs
    return (
        item.command.return_type == "VkResult"
        and item.output.vector is None
        and len(outputs) >= 2
        and all(
            output.type != "void"
            and (
                not output.length
                or _output_size_expression(output, item.command, registry) is not None
            )
            for output in outputs
        )
    )


def _multi_output_decl(
    item: CommandAnalysis,
    receiver: str | None,
    registry: Registry,
    config: GeneratorConfig,
) -> str | None:
    parts = _multi_output_parts(item, receiver, registry, config)
    if parts is None:
        return None
    result, params, _ = parts
    return f"    [[nodiscard]] {result} {_callable_name(item, receiver, config)}({params}) const;"


def _multi_output_impl(
    item: CommandAnalysis,
    receiver: str | None,
    registry: Registry,
    config: GeneratorConfig,
) -> str | None:
    parts = _multi_output_parts(item, receiver, registry, config)
    if parts is None:
        return None
    result, params, body = parts
    receiver_name = (
        _cpp_type(receiver, registry, config) if receiver is not None else "Context"
    )
    return f"inline {result} {receiver_name}::{_callable_name(item, receiver, config)}({params}) const {{ {body} }}"


def _emit_handle(
    item: HandleAnalysis,
    analysis: ApiAnalysis,
    registry: Registry,
    config: GeneratorConfig,
    injection: list[str],
    vma_resources: frozenset[str],
) -> str:
    h = item.type
    name = item.cpp_name
    parent = _cpp_type(h.parent.split(",")[0], registry, config) if h.parent else None
    state_name = f"{name}ControlBlock"
    state_lines = [
        f"namespace detail {{ struct {state_name} final : LifetimeHeader {{",
        f"    using native_type = {h.name};",
        # Serializes lookup/retain against detach/final release.  Host-tracked
        # types also use it for their map; device-scoped types have no map and
        # use Vulkan private data as the association store.
        "    inline static std::shared_mutex tracking_mutex;",
        "    native_type native{};",
    ]
    uses_host_registry = not _is_device_scope(h, registry) or h.name == "VkDevice"
    if uses_host_registry:
        state_lines.append(
            f"    inline static std::unordered_multimap<std::uint64_t, {state_name}*> registry;"
        )
    if parent:
        state_lines.append(f"    {parent} parent{{}};")
    state_lines.extend(
        [
            "    std::unique_ptr<std::unordered_map<std::type_index, std::shared_ptr<const void>>> data;",
            f"    std::function<void({'const ' + parent + '&, ' if parent else ''}native_type)> destroyer;",
        ]
    )
    if h.name == "VkInstance":
        state_lines.append("    VolkInstanceTable instance_dispatch{};")
    elif h.name == "VkDevice":
        state_lines.extend(
            [
                "    std::shared_mutex private_data_mutex;",
                "    DeviceAssociation device_association{};",
                "    VolkDeviceTable device_dispatch{};",
            ]
        )
    if item.create_info:
        state_lines.append(
            f"    std::shared_ptr<const {_cpp_type(item.create_info, registry, config)}> create_info;"
        )
    vma_resource = h.name in vma_resources
    if vma_resource:
        state_lines.extend(
            [
                "    std::shared_ptr<void> vma_allocator_lifetime;",
                "    VmaAllocator vma_allocator{};",
                "    VmaAllocation vma_allocation{};",
                "    VmaAllocationInfo vma_allocation_info{};",
                "    VmaAllocationCreateInfo vma_allocation_create_info{};",
            ]
        )
    state_lines.extend(
        [
            f"    static void detach({state_name}* self) noexcept;",
            f"    static void finalize({state_name}* self) noexcept;",
            "}; }",
        ]
    )
    lines = [
        "\n".join(state_lines),
        f"class {name} {{",
        "  public:",
        f"    using native_type = {h.name};",
        "  private:",
        "    native_type native_{};",
    ]
    if parent:
        lines.append(f"    mutable {parent} parent_{{}};")
    else:
        lines.append("    VolkInstanceTable dispatch_{};")
    lines.append(f"    mutable detail::{state_name}* ctrl_{{}};")
    lines.append("    friend struct detail::ExternsyncAccess;")
    lines.append(
        "    template <typename Handle> friend bool detail::same_object(const Handle&, const Handle&) noexcept;"
    )
    releasers = handle_releasers(registry, {handle.name for handle in registry.handles})
    producer_receivers: set[str | None] = set()
    for command in analysis.commands:
        if any(
            output.type == h.name
            and is_owned_handle_output(command.command, output, releasers)
            for output in command.output.outputs
        ):
            producer_receivers.update(command.receivers or (None,))
    for producer_receiver in sorted(producer_receivers, key=lambda value: value or ""):
        if producer_receiver is None:
            lines.append("    friend class Context;")
        elif producer_receiver in analysis.handles:
            lines.append(
                f"    friend class {_cpp_type(producer_receiver, registry, config)};"
            )
    lines.append(f"    explicit {name}(detail::{state_name}* state) noexcept;")
    borrowed_arguments = (
        f"native_type native, {parent} parent" if parent else "native_type native"
    )
    lines.append(f"    explicit {name}({borrowed_arguments}) noexcept;")
    factory_parent_arg = f", const {parent}& parent" if parent else ""
    factory_destroyer = (
        f"std::function<void({'const ' + parent + '&, ' if parent else ''}native_type)>"
    )
    factory_record_arg = ""
    if item.create_info:
        factory_record_arg = f", std::shared_ptr<const {_cpp_type(item.create_info, registry, config)}> creationRecord"
    lines.append(
        f"    [[nodiscard]] static Result<{name}> makeOwned(native_type native{factory_parent_arg}, "
        f"{factory_destroyer} destroyer{factory_record_arg});"
    )
    lines.extend(
        [
            "  public:",
            f"    {name}() noexcept;",
            f"    {name}(std::nullptr_t) noexcept;",
            f"    ~{name}();",
            f"    {name}(const {name}& other) noexcept;",
            f"    {name}({name}&& other) noexcept;",
            f"    {name}& operator=({name} other) noexcept;",
            f"    void swap({name}& other) noexcept;",
            "    void reset() noexcept;",
            "    [[nodiscard]] native_type raw() const noexcept { return native_; }",
            "    [[nodiscard]] long use_count() const noexcept { return ctrl_ ? static_cast<long>(ctrl_->refs.load(std::memory_order_acquire)) : 0; }",
            "    [[nodiscard]] std::uintptr_t id() const noexcept { return ctrl_ ? reinterpret_cast<std::uintptr_t>(ctrl_) : detail::raw_key(native_); }",
            "    [[nodiscard]] explicit operator bool() const noexcept { return raw() != native_type{}; }",
            f"    [[nodiscard]] bool sameNativeHandle(const {name}& rhs) const noexcept {{ return raw() == rhs.raw(); }}",
            f"    friend bool operator==(const {name}& lhs, const {name}& rhs) noexcept {{ return lhs.ctrl_ == rhs.ctrl_ && lhs.raw() == rhs.raw(); }}",
        ]
    )
    if h.name == "VkDevice":
        association_expr = (
            "ctrl_ ? ctrl_->device_association : detail::DeviceAssociation{}"
        )
        dispatch_expr = "detail::DispatchState{parent().dispatchState().instance, ctrl_ ? &ctrl_->device_dispatch : nullptr, native_}"
    elif h.name == "VkInstance":
        association_expr = "detail::DeviceAssociation{}"
        dispatch_expr = "detail::DispatchState{ctrl_ ? &ctrl_->instance_dispatch : &dispatch_, nullptr, {}}"
    else:
        association_expr = (
            "parent().deviceAssociation()"
            if parent and _is_device_scope(h, registry)
            else "detail::DeviceAssociation{}"
        )
        dispatch_expr = "parent().dispatchState()" if parent else "dispatch_"
    lines.append(
        f"    [[nodiscard]] detail::DeviceAssociation deviceAssociation() const noexcept {{ return {association_expr}; }}"
    )
    lines.append(
        f"    [[nodiscard]] detail::DispatchState dispatchState() const noexcept {{ return {dispatch_expr}; }}"
    )
    if parent:
        lines.append(
            f"    [[nodiscard]] const {parent}& parent() const noexcept {{ return ctrl_ ? ctrl_->parent : parent_; }}"
        )
    if item.create_info:
        cpp_info = _cpp_type(item.create_info, registry, config)
        lines.append(
            f"    [[nodiscard]] const {cpp_info}* createInfo() const noexcept {{ return ctrl_ ? ctrl_->create_info.get() : nullptr; }}"
        )
    if vma_resource:
        lines.append(
            "    [[nodiscard]] VmaAllocation allocation() const noexcept { return ctrl_ ? ctrl_->vma_allocation : VmaAllocation{}; }"
        )
        lines.append(
            "    [[nodiscard]] const VmaAllocationInfo* allocationInfo() const noexcept { return ctrl_ && ctrl_->vma_allocation != VmaAllocation{} ? &ctrl_->vma_allocation_info : nullptr; }"
        )
        lines.append(
            "    [[nodiscard]] const VmaAllocationCreateInfo* allocationCreateInfo() const noexcept { return ctrl_ && ctrl_->vma_allocation != VmaAllocation{} ? &ctrl_->vma_allocation_create_info : nullptr; }"
        )
    lines.extend(
        [
            "    template <typename T> [[nodiscard]] Result<void> setData(std::shared_ptr<const T> value) const;",
            "    template <typename T> [[nodiscard]] std::shared_ptr<const T> getData() const noexcept;",
            "    template <typename T> void clearData() const noexcept;",
        ]
    )
    adoption = f", const {parent}& parent" if parent else ""
    lines.append(
        f"    [[nodiscard]] static Result<{name}> borrow(native_type native{adoption});"
    )
    create_info_arg = ""
    if item.create_info:
        cpp_info = _cpp_type(item.create_info, registry, config)
        create_info_arg = f", std::shared_ptr<const {cpp_info}> creationRecord = {{}}"
    lines.append(
        f"    [[nodiscard]] static Result<{name}> adopt(native_type native{adoption}, std::function<void(native_type)> destroyer{create_info_arg});"
    )
    if vma_resource and parent:
        create_record = (
            f", std::shared_ptr<const {_cpp_type(item.create_info, registry, config)}> creationRecord"
            if item.create_info
            else ""
        )
        lines.append(
            f"    [[nodiscard]] static Result<{name}> adoptVma(native_type native, const {parent}& parent, std::shared_ptr<void> allocatorLifetime, VmaAllocator allocator, VmaAllocation allocation, const VmaAllocationInfo& allocationInfo, const VmaAllocationCreateInfo& allocationCreateInfo{create_record});"
        )
    seen: set[tuple[str, str]] = set()
    for command in item.commands:
        declaration = _method_decl(command, h.name, registry, config)
        method_name = _method_name(command, h.name, config)
        key = (method_name, declaration)
        if key not in seen:
            lines.append(
                _guard(
                    declaration,
                    command.command.protect or command.command.availability.protect,
                )
            )
            seen.add(key)
        convenience = _convenience_decl(command, h.name, registry, config)
        if convenience:
            key = (method_name + "#convenience", convenience)
            if key not in seen:
                lines.append(
                    _guard(
                        convenience,
                        command.command.protect or command.command.availability.protect,
                    )
                )
                seen.add(key)
        owned_convenience = _owned_handle_convenience_decl(
            command, h.name, registry, config
        )
        if owned_convenience:
            key = (method_name + "#owned", owned_convenience)
            if key not in seen:
                lines.append(
                    _guard(
                        owned_convenience,
                        command.command.protect or command.command.availability.protect,
                    )
                )
                seen.add(key)
        multi_output = _multi_output_decl(command, h.name, registry, config)
        if multi_output:
            key = (method_name + "#multi", multi_output)
            if key not in seen:
                lines.append(
                    _guard(
                        multi_output,
                        command.command.protect or command.command.availability.protect,
                    )
                )
                seen.add(key)
    lines.extend(line.rstrip("\r\n") for line in injection)
    lines.append("};")
    return _guard("\n".join(lines), h.protect or h.availability.protect)


def _emit_handles(
    analysis: ApiAnalysis,
    registry: Registry,
    config: GeneratorConfig,
    template: Template,
    vma_resources: frozenset[str],
) -> str:
    # Parent wrapper definitions must precede children because parent(),
    # borrow(), and adopt() are inline and call the parent's public API.
    pending = list(analysis.handles.values())
    ordered: list[HandleAnalysis] = []
    emitted: set[str] = set()
    while pending:
        progress = False
        for item in list(pending):
            parent = item.type.parent.split(",")[0] if item.type.parent else None
            if parent not in analysis.handles or parent in emitted:
                ordered.append(item)
                emitted.add(item.type.name)
                pending.remove(item)
                progress = True
        if not progress:
            ordered.extend(pending)
            break
    handles = "\n\n".join(
        _emit_handle(
            item,
            analysis,
            registry,
            config,
            template.injections.get(item.cpp_name, []),
            vma_resources,
        )
        for item in ordered
    )
    return handles + "\n\n" + _emit_context(analysis, registry, config)


def _emit_handle_lifetime_impl(
    item: HandleAnalysis,
    registry: Registry,
    config: GeneratorConfig,
    vma_resources: frozenset[str],
) -> str:
    h = item.type
    name = item.cpp_name
    state_name = f"{name}ControlBlock"
    parent = _cpp_type(h.parent.split(",")[0], registry, config) if h.parent else None
    object_type = h.object_type_enum or "VK_OBJECT_TYPE_UNKNOWN"
    device_scope = _is_device_scope(h, registry)
    vma_resource = h.name in vma_resources
    uses_host_registry = not device_scope or h.name == "VkDevice"
    lines = [
        f"inline void detail::{state_name}::detach(detail::{state_name}* self) noexcept {{"
    ]
    if device_scope:
        association = (
            "self->device_association"
            if h.name == "VkDevice"
            else "self->parent.deviceAssociation()"
        )
        lines.extend(
            [
                f"    auto association = {association};",
                f"    if (association && {object_type} != VK_OBJECT_TYPE_UNKNOWN) {{",
                "        std::unique_lock association_lock(*association.mutex);",
                f"        auto status = association.dispatch->vkSetPrivateData(association.device, {object_type}, detail::raw_key(self->native), association.slot, 0);",
                f'        if (status != VK_SUCCESS) detail::report_error(static_cast<ResultCode>(status), "{name}", detail::raw_key(self->native));',
                "    }",
            ]
        )
    if uses_host_registry:
        lines.extend(
            [
                "    auto [first, last] = registry.equal_range(detail::raw_key(self->native));",
                "    for (auto found = first; found != last; ++found) if (found->second == self) { registry.erase(found); break; }",
            ]
        )
    lines.append("}")
    lines.extend(
        [
            f"inline void detail::{state_name}::finalize(detail::{state_name}* self) noexcept {{",
            "    try {",
        ]
    )
    lines.append("        std::unique_lock access(self->externsync);")
    if h.name == "VkDevice":
        lines.extend(
            [
                "        if (self->device_association) {",
                "            self->device_dispatch.vkDestroyPrivateDataSlot(self->device_association.device, self->device_association.slot, nullptr);",
                "            self->device_association = {};",
                "        }",
            ]
        )
    if vma_resource:
        function = "vmaDestroyBuffer" if h.name == "VkBuffer" else "vmaDestroyImage"
        destroy_call = (
            "self->destroyer(self->parent, self->native)"
            if parent
            else "self->destroyer(self->native)"
        )
        lines.extend(
            [
                f"        if (self->vma_allocation != VmaAllocation{{}}) {{ {function}(self->vma_allocator, self->native, self->vma_allocation); self->vma_allocation = {{}}; }}",
                f"        else if (self->destroyer && self->native != native_type{{}}) {destroy_call};",
            ]
        )
    else:
        destroy_call = (
            "self->destroyer(self->parent, self->native)"
            if parent
            else "self->destroyer(self->native)"
        )
        lines.append(
            f"        if (self->destroyer && self->native != native_type{{}}) {destroy_call};"
        )
    lines.extend(
        [
            f'    }} catch (...) {{ detail::report_error(ResultCode::ErrorUnknown, "{name}", detail::raw_key(self->native)); }}',
            "    self->native = {};",
            "    delete self;",
            "}",
        ]
    )

    managed_initializers = "native_(state ? state->native : native_type{})"
    if parent:
        managed_initializers += ", parent_{}"
    elif h.name == "VkInstance":
        managed_initializers += ", dispatch_{}"
    else:
        managed_initializers += ", dispatch_{}"
    managed_initializers += ", ctrl_(state)"
    lines.append(
        f"inline {name}::{name}(detail::{state_name}* state) noexcept : {managed_initializers} {{}}"
    )
    borrowed_params = (
        f"native_type native, {parent} parent" if parent else "native_type native"
    )
    borrowed_initializers = (
        "native_(native), parent_(std::move(parent))" if parent else "native_(native)"
    )
    lines.append(
        f"inline {name}::{name}({borrowed_params}) noexcept : {borrowed_initializers} {{}}"
    )

    copy_initializers = (
        "native_(other.native_)"
        + (
            f", parent_(other.ctrl_ ? {parent}{{}} : other.parent_)"
            if parent
            else ", dispatch_(other.dispatch_)"
        )
        + ", ctrl_(other.ctrl_)"
    )
    move_initializers = (
        "native_(std::exchange(other.native_, native_type{}))"
        + (
            ", parent_(std::move(other.parent_))"
            if parent
            else ", dispatch_(other.dispatch_)"
        )
        + ", ctrl_(std::exchange(other.ctrl_, nullptr))"
    )
    lines.extend(
        [
            f"inline {name}::{name}() noexcept = default;",
            f"inline {name}::{name}(std::nullptr_t) noexcept {{}}",
            f"inline {name}::~{name}() {{ reset(); }}",
            f"inline {name}::{name}(const {name}& other) noexcept : {copy_initializers} {{ if (ctrl_) ctrl_->retain(); }}",
            f"inline {name}::{name}({name}&& other) noexcept : {move_initializers} {{}}",
            f"inline {name}& {name}::operator=({name} other) noexcept {{ swap(other); return *this; }}",
        ]
    )
    swap_tail = (
        "parent_.swap(other.parent_);"
        if parent
        else "std::swap(dispatch_, other.dispatch_);"
    )
    reset_tail = "parent_ = {};" if parent else "dispatch_ = {};"
    lines.extend(
        [
            f"inline void {name}::swap({name}& other) noexcept {{ std::swap(native_, other.native_); std::swap(ctrl_, other.ctrl_); {swap_tail} }}",
            f"inline void {name}::reset() noexcept {{ auto* state = std::exchange(ctrl_, nullptr); native_ = {{}}; {reset_tail} if (state) state->release(detail::{state_name}::tracking_mutex, state, &detail::{state_name}::detach, &detail::{state_name}::finalize); }}",
        ]
    )

    adoption = f", const {parent}& parent" if parent else ""
    borrowed_ctor = f"{name}(native, parent)" if parent else f"{name}(native)"
    if h.name == "VkDevice":
        borrow_body = f"if (native == native_type{{}}) return std::unexpected(ResultCode::ErrorUnknown); std::shared_lock lock(detail::{state_name}::tracking_mutex); auto [first, last] = detail::{state_name}::registry.equal_range(detail::raw_key(native)); for (auto found = first; found != last; ++found) if (detail::same_object(found->second->parent, parent)) {{ found->second->retain(); return {name}(found->second); }} lock.unlock(); return {borrowed_ctor};"
    else:
        if device_scope:
            lookup = f"auto association = parent.deviceAssociation(); if (association && {object_type} != VK_OBJECT_TYPE_UNKNOWN) {{ std::shared_lock lock(detail::{state_name}::tracking_mutex); std::shared_lock association_lock(*association.mutex); std::uint64_t existing{{}}; association.dispatch->vkGetPrivateData(association.device, {object_type}, detail::raw_key(native), association.slot, &existing); if (existing) {{ auto* state = static_cast<detail::{state_name}*>(reinterpret_cast<detail::LifetimeHeader*>(static_cast<std::uintptr_t>(existing))); state->retain(); return {name}(state); }} }}"
        else:
            parent_filter = (
                f" && detail::same_object(found->second->parent, parent)"
                if parent
                else ""
            )
            lookup = f"std::shared_lock lock(detail::{state_name}::tracking_mutex); auto [first, last] = detail::{state_name}::registry.equal_range(detail::raw_key(native)); for (auto found = first; found != last; ++found) if (found->second{parent_filter}) {{ found->second->retain(); return {name}(found->second); }} lock.unlock();"
        borrow_body = f"if (native == native_type{{}}) return std::unexpected(ResultCode::ErrorUnknown); {lookup} auto value = {borrowed_ctor};"
        if h.name == "VkInstance":
            borrow_body += " volkLoadInstanceTable(&value.dispatch_, native);"
        borrow_body += " return value;"
    lines.append(
        f"inline Result<{name}> {name}::borrow(native_type native{adoption}) {{ {borrow_body} }}"
    )
    create_info_arg = ""
    if item.create_info:
        cpp_info = _cpp_type(item.create_info, registry, config)
        create_info_arg = f", std::shared_ptr<const {cpp_info}> creationRecord"
    destroyer_type = (
        f"std::function<void({'const ' + parent + '&, ' if parent else ''}native_type)>"
    )
    offered_destroy = "destroyer(parent, native)" if parent else "destroyer(native)"
    factory_parent_arg = f", const {parent}& parent" if parent else ""
    # makeOwned publishes a fresh, untracked native handle into a new owning
    # block.  It performs no lookup, so create/allocate commands pay for only
    # allocation plus association publication.
    make_lines = [
        f"inline Result<{name}> {name}::makeOwned(native_type native{factory_parent_arg}, {destroyer_type} destroyer{create_info_arg}) {{",
        "    if (!destroyer || native == native_type{}) return std::unexpected(ResultCode::ErrorUnknown);",
    ]
    if device_scope and h.name != "VkDevice":
        make_lines.extend(
            [
                "    auto association = parent.deviceAssociation();",
                f"    if (!association || {object_type} == VK_OBJECT_TYPE_UNKNOWN) {{ {offered_destroy}; return std::unexpected(ResultCode::ErrorUnknown); }}",
                "    std::unique_lock association_lock(*association.mutex);",
                f"    auto* state = new (std::nothrow) detail::{state_name};",
                f"    if (!state) {{ association_lock.unlock(); {offered_destroy}; return std::unexpected(ResultCode::ErrorOutOfHostMemory); }}",
                "    state->native = native;",
                "    state->parent = parent;",
                f"    auto status = association.dispatch->vkSetPrivateData(association.device, {object_type}, detail::raw_key(native), association.slot, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(static_cast<detail::LifetimeHeader*>(state))));",
                f"    if (status != VK_SUCCESS) {{ association_lock.unlock(); delete state; {offered_destroy}; return std::unexpected(static_cast<ResultCode>(status)); }}",
            ]
        )
    else:
        make_lines.extend(
            [
                f"    std::unique_lock lock(detail::{state_name}::tracking_mutex);",
                f"    auto* state = new (std::nothrow) detail::{state_name};",
                f"    if (!state) {{ lock.unlock(); {offered_destroy}; return std::unexpected(ResultCode::ErrorOutOfHostMemory); }}",
                "    state->native = native;",
            ]
        )
        if parent:
            make_lines.append("    state->parent = parent;")
        if h.name == "VkInstance":
            make_lines.append(
                "    volkLoadInstanceTable(&state->instance_dispatch, native);"
            )
        if h.name == "VkDevice":
            make_lines.extend(
                [
                    "    volkLoadDeviceTable(&state->device_dispatch, native);",
                    "    state->device_association.device = native;",
                    "    state->device_association.dispatch = &state->device_dispatch;",
                    "    state->device_association.mutex = &state->private_data_mutex;",
                    "    VkPrivateDataSlotCreateInfo info{VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO};",
                    "    auto setup_status = state->device_dispatch.vkCreatePrivateDataSlot(native, &info, nullptr, &state->device_association.slot);",
                    f"    if (setup_status != VK_SUCCESS) {{ delete state; lock.unlock(); {offered_destroy}; return std::unexpected(static_cast<ResultCode>(setup_status)); }}",
                    "    setup_status = state->device_dispatch.vkSetPrivateData(native, VK_OBJECT_TYPE_DEVICE, detail::raw_key(native), state->device_association.slot, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(static_cast<detail::LifetimeHeader*>(state))));",
                    f"    if (setup_status != VK_SUCCESS) {{ state->device_dispatch.vkDestroyPrivateDataSlot(native, state->device_association.slot, nullptr); delete state; lock.unlock(); {offered_destroy}; return std::unexpected(static_cast<ResultCode>(setup_status)); }}",
                ]
            )
        make_lines.append(
            f"    detail::{state_name}::registry.emplace(detail::raw_key(native), state);"
        )
    make_lines.append("    state->destroyer = std::move(destroyer);")
    if item.create_info:
        make_lines.append("    state->create_info = std::move(creationRecord);")
    make_lines.extend([f"    return {name}(state);", "}"])
    lines.append("\n".join(make_lines))

    # adopt: public ownership transfer.  Reuse an existing block when the native
    # handle is already tracked; otherwise create an owning block.
    adopt_lines = [
        f"inline Result<{name}> {name}::adopt(native_type native{adoption}, std::function<void(native_type)> destroyer{create_info_arg}) {{",
        "    if (!destroyer || native == native_type{}) return std::unexpected(ResultCode::ErrorUnknown);",
    ]
    if device_scope and h.name != "VkDevice":
        adopt_lines.extend(
            [
                "    auto association = parent.deviceAssociation();",
                f"    if (association && {object_type} != VK_OBJECT_TYPE_UNKNOWN) {{",
                f"        std::shared_lock lock(detail::{state_name}::tracking_mutex);",
                "        std::shared_lock association_lock(*association.mutex);",
                "        std::uint64_t existing{};",
                f"        association.dispatch->vkGetPrivateData(association.device, {object_type}, detail::raw_key(native), association.slot, &existing);",
                f"        if (existing) {{ auto* state = static_cast<detail::{state_name}*>(reinterpret_cast<detail::LifetimeHeader*>(static_cast<std::uintptr_t>(existing))); state->retain(); return {name}(state); }}",
                "    }",
            ]
        )
    else:
        parent_filter = (
            f" && detail::same_object(found->second->parent, parent)" if parent else ""
        )
        adopt_lines.extend(
            [
                f"    std::shared_lock lock(detail::{state_name}::tracking_mutex);",
                f"    auto [first, last] = detail::{state_name}::registry.equal_range(detail::raw_key(native));",
                f"    for (auto found = first; found != last; ++found) if (found->second{parent_filter}) {{ found->second->retain(); return {name}(found->second); }}",
                "    lock.unlock();",
            ]
        )
    if parent:
        adapter = f", [destroyer = std::move(destroyer)](const {parent}&, native_type value) {{ destroyer(value); }}"
    else:
        adapter = ", std::move(destroyer)"
    adopt_record = ", std::move(creationRecord)" if item.create_info else ""
    parent_arg = ", parent" if parent else ""
    adopt_lines.append(
        f"    return makeOwned(native{parent_arg}{adapter}{adopt_record});"
    )
    adopt_lines.append("}")
    lines.append("\n".join(adopt_lines))
    if vma_resource and parent:
        vma_destroy = "vmaDestroyBuffer" if h.name == "VkBuffer" else "vmaDestroyImage"
        create_record_arg = (
            f", std::shared_ptr<const {_cpp_type(item.create_info, registry, config)}> creationRecord"
            if item.create_info
            else ""
        )
        create_record_store = (
            " state->create_info = std::move(creationRecord);"
            if item.create_info
            else ""
        )
        lines.append(
            f"inline Result<{name}> {name}::adoptVma(native_type native, const {parent}& parent, std::shared_ptr<void> allocatorLifetime, VmaAllocator allocator, VmaAllocation allocation, const VmaAllocationInfo& allocationInfo, const VmaAllocationCreateInfo& allocationCreateInfo{create_record_arg}) {{ if (!allocator || !allocation || native == native_type{{}}) return std::unexpected(ResultCode::ErrorUnknown); auto association = parent.deviceAssociation(); if (!association) {{ {vma_destroy}(allocator, native, allocation); return std::unexpected(ResultCode::ErrorUnknown); }} std::unique_lock lock(detail::{state_name}::tracking_mutex); std::unique_lock association_lock(*association.mutex); std::uint64_t existing{{}}; association.dispatch->vkGetPrivateData(association.device, {object_type}, detail::raw_key(native), association.slot, &existing); detail::{state_name}* state = existing ? static_cast<detail::{state_name}*>(reinterpret_cast<detail::LifetimeHeader*>(static_cast<std::uintptr_t>(existing))) : nullptr; if (state) {{ state->retain(); return {name}(state); }} state = new (std::nothrow) detail::{state_name}; if (!state) {{ association_lock.unlock(); lock.unlock(); {vma_destroy}(allocator, native, allocation); return std::unexpected(ResultCode::ErrorOutOfHostMemory); }} state->native = native; state->parent = parent; state->vma_allocator_lifetime = std::move(allocatorLifetime); state->vma_allocator = allocator; state->vma_allocation = allocation; state->vma_allocation_info = allocationInfo; state->vma_allocation_create_info = allocationCreateInfo;{create_record_store} auto status = association.dispatch->vkSetPrivateData(association.device, {object_type}, detail::raw_key(native), association.slot, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(static_cast<detail::LifetimeHeader*>(state)))); if (status != VK_SUCCESS) {{ association_lock.unlock(); delete state; lock.unlock(); {vma_destroy}(allocator, native, allocation); return std::unexpected(static_cast<ResultCode>(status)); }} return {name}(state); }}"
        )
    return _guard("\n".join(lines), h.protect or h.availability.protect)


def _emit_handle_implementations(
    analysis: ApiAnalysis,
    registry: Registry,
    config: GeneratorConfig,
    vma_resources: frozenset[str],
) -> str:
    result: list[str] = []
    for handle in analysis.handles.values():
        result.append(
            _emit_handle_lifetime_impl(handle, registry, config, vma_resources)
        )
        seen: set[str] = set()
        for command in handle.commands:
            implementation = _method_impl(command, handle.type.name, registry, config)
            if implementation and implementation not in seen:
                result.append(
                    _guard(
                        implementation,
                        command.command.protect or command.command.availability.protect,
                    )
                )
                seen.add(implementation)
            convenience = _convenience_impl(command, handle.type.name, registry, config)
            if convenience and convenience not in seen:
                result.append(
                    _guard(
                        convenience,
                        command.command.protect or command.command.availability.protect,
                    )
                )
                seen.add(convenience)
            owned_convenience = _owned_handle_convenience_impl(
                command, handle.type.name, registry, config
            )
            if owned_convenience and owned_convenience not in seen:
                result.append(
                    _guard(
                        owned_convenience,
                        command.command.protect or command.command.availability.protect,
                    )
                )
                seen.add(owned_convenience)
            multi_output = _multi_output_impl(
                command, handle.type.name, registry, config
            )
            if multi_output and multi_output not in seen:
                result.append(
                    _guard(
                        multi_output,
                        command.command.protect or command.command.availability.protect,
                    )
                )
                seen.add(multi_output)
    result.append(_emit_context_implementations(analysis, registry, config))
    return "\n\n".join(result)


def _emit_handle_template_implementations(
    analysis: ApiAnalysis, registry: Registry, config: GeneratorConfig
) -> str:
    result: list[str] = []
    for item in analysis.handles.values():
        name = item.cpp_name
        definitions = [
            f"template <typename T> inline Result<void> {name}::setData(std::shared_ptr<const T> value) const {{ if (!ctrl_) return std::unexpected(ResultCode::ErrorUnknown); std::unique_lock lock(ctrl_->externsync); try {{ if (!ctrl_->data) ctrl_->data = std::make_unique<std::unordered_map<std::type_index, std::shared_ptr<const void>>>(); ctrl_->data->insert_or_assign(typeid(T), std::move(value)); }} catch (...) {{ return std::unexpected(ResultCode::ErrorOutOfHostMemory); }} return {{}}; }}",
            f"template <typename T> inline std::shared_ptr<const T> {name}::getData() const noexcept {{ if (!ctrl_) return nullptr; std::shared_lock lock(ctrl_->externsync); if (!ctrl_->data) return nullptr; auto found = ctrl_->data->find(typeid(T)); return found == ctrl_->data->end() ? nullptr : std::static_pointer_cast<const T>(found->second); }}",
            f"template <typename T> inline void {name}::clearData() const noexcept {{ if (!ctrl_) return; std::unique_lock lock(ctrl_->externsync); if (ctrl_->data) ctrl_->data->erase(typeid(T)); }}",
        ]
        result.append(
            _guard(
                "\n\n".join(definitions),
                item.type.protect or item.type.availability.protect,
            )
        )
    return "\n\n".join(result)


def _emit_forwards(registry: Registry, config: GeneratorConfig) -> str:
    values = []
    for item in registry.structs:
        if (
            not item.alias
            and item.category == "struct"
            and _has_native_definition(item)
        ):
            values.append(f"struct {_cpp_type(item.name, registry, config)};")
    for item in registry.handles:
        if not item.alias:
            values.append(f"class {_cpp_type(item.name, registry, config)};")
            if len(creation_infos_for_handle(registry, item)) > 1:
                values.append(
                    f"struct {_cpp_type(item.name, registry, config)}CreationRecord;"
                )
    return "\n".join(values)


def _emit_command_result_forwards(analysis: ApiAnalysis, registry: Registry) -> str:
    return "\n".join(
        f"struct {_command_result_name(item)};"
        for item in analysis.commands
        if _has_multi_output_result(item, registry)
    )


def _emit_extensions(registry: Registry, config: GeneratorConfig) -> str:
    lines = []
    for item in registry.structs:
        extension = _cpp_type(item.name, registry, config)
        for base in item.struct_extends:
            lines.append(
                f"template <> struct StructureExtends<{_cpp_type(base, registry, config)}, {extension}> : std::true_type {{}};"
            )
    return "\n".join(lines)


def _context_method_parts(
    item: CommandAnalysis, registry: Registry, config: GeneratorConfig
) -> tuple[str, str, str]:
    return _command_parts(item, None, registry, config)


def _emit_context(
    analysis: ApiAnalysis, registry: Registry, config: GeneratorConfig
) -> str:
    commands = [item for item in analysis.commands if not item.receivers]
    version = config.minimum_core.replace(".", "_")
    lines = [
        "// Receiver-less API owner; Context is deliberately not a handle.",
        "class Context {",
        "    Context() noexcept = default;",
        "  public:",
        f"    static constexpr std::uint32_t minimumApiVersion = VK_API_VERSION_{version};",
        "    [[nodiscard]] static Result<Context> create();",
    ]
    for item in commands:
        result, params, _ = _context_method_parts(item, registry, config)
        prefix = "" if result == "void" else "[[nodiscard]] "
        lines.append(f"    {prefix}{result} {item.cpp_name}({params}) const;")
        convenience = _convenience_decl(item, None, registry, config)
        if convenience:
            lines.append(convenience)
        value_convenience = _owned_handle_convenience_decl(item, None, registry, config)
        if value_convenience:
            lines.append(value_convenience)
        multi_output = _multi_output_decl(item, None, registry, config)
        if multi_output:
            lines.append(multi_output)
    lines.append("};")
    return "\n".join(lines)


def _emit_context_implementations(
    analysis: ApiAnalysis, registry: Registry, config: GeneratorConfig
) -> str:
    lines: list[str] = [
        "inline Result<Context> Context::create() {",
        "    auto status = volkInitialize();",
        "    if (status != VK_SUCCESS) return std::unexpected(static_cast<ResultCode>(status));",
        "    if (volkGetInstanceVersion() < minimumApiVersion) return std::unexpected(static_cast<ResultCode>(VK_ERROR_INCOMPATIBLE_DRIVER));",
        "    return Context{};",
        "}",
    ]
    for item in analysis.commands:
        if item.receivers:
            continue
        result, params, body = _context_method_parts(item, registry, config)
        if body:
            lines.append(
                _guard(
                    f"inline {result} Context::{item.cpp_name}({params}) const {{ {body} }}",
                    item.command.protect or item.command.availability.protect,
                )
            )
        convenience = _convenience_impl(item, None, registry, config)
        if convenience:
            lines.append(
                _guard(
                    convenience,
                    item.command.protect or item.command.availability.protect,
                )
            )
        value_convenience = _owned_handle_convenience_impl(item, None, registry, config)
        if value_convenience:
            lines.append(
                _guard(
                    value_convenience,
                    item.command.protect or item.command.availability.protect,
                )
            )
        multi_output = _multi_output_impl(item, None, registry, config)
        if multi_output:
            lines.append(
                _guard(
                    multi_output,
                    item.command.protect or item.command.availability.protect,
                )
            )
    return "\n\n".join(lines)


def _emit_command_metadata(
    analysis: ApiAnalysis, registry: Registry, config: GeneratorConfig
) -> str:
    lines = [
        "// Generated multi-output records and command metadata retained for custom lowering."
    ]
    for item in analysis.commands:
        if not _has_multi_output_result(item, registry):
            continue
        lines.append(f"struct {_command_result_name(item)} {{")
        for output in item.output.outputs:
            cpp = _cpp_type(output.type, registry, config)
            field_type = f"std::vector<{cpp}>" if output.length else cpp
            lines.append(f"    {field_type} {_public_param_name(output)}{{}};")
        lines.append("};")
    for item in analysis.commands:
        successes = ",".join(item.command.success_codes)
        receivers = ",".join(item.receivers) or "Context"
        outputs = ",".join(output.name for output in item.output.outputs)
        lines.append(
            f"// {item.command.name}: receivers={receivers}; success={successes}; outputs={outputs}; externsync={','.join(p.name for p in item.command.params if p.externsync)}"
        )
    return "\n".join(lines)


RUNTIME = r"""namespace detail {
struct LifetimeHeader {
    std::atomic_uint64_t refs{1};
    mutable std::shared_mutex externsync;
    void retain() noexcept { refs.fetch_add(1, std::memory_order_relaxed); }
    template <typename ControlBlock>
    void release(std::shared_mutex& mutex, ControlBlock* self, void (*detach)(ControlBlock*) noexcept, void (*finalize)(ControlBlock*) noexcept) noexcept {
        std::unique_lock lock(mutex);
        if (refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
        detach(self);
        lock.unlock();
        finalize(self);
    }
};
struct DeviceAssociation {
    VkDevice device{};
    VkPrivateDataSlot slot{};
    const VolkDeviceTable* dispatch{};
    std::shared_mutex* mutex{};
    [[nodiscard]] explicit operator bool() const noexcept { return device != VkDevice{} && slot != VkPrivateDataSlot{} && dispatch != nullptr && mutex != nullptr; }
};
struct DispatchState {
    const VolkInstanceTable* instance{};
    const VolkDeviceTable* device{};
    VkDevice native_device{};
};
using ErrorSink = void (*)(ResultCode, std::string_view, std::uint64_t) noexcept;
inline void default_error_sink(ResultCode, std::string_view, std::uint64_t) noexcept {}
inline std::atomic<ErrorSink> error_sink{default_error_sink};
inline void set_error_sink(ErrorSink sink) noexcept { error_sink.store(sink ? sink : default_error_sink, std::memory_order_release); }
inline void report_error(ResultCode result, std::string_view type = {}, std::uint64_t identity = 0) noexcept { error_sink.load(std::memory_order_acquire)(result, type, identity); }
template <typename Native> [[nodiscard]] std::uint64_t raw_key(Native value) noexcept {
    if constexpr (std::is_pointer_v<Native>) return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(value));
    else return static_cast<std::uint64_t>(value);
}
struct StateLockRef {
    std::uintptr_t identity{};
    std::shared_mutex* mutex{};
    bool exclusive{};
    DeviceAssociation association{};
    VkObjectType object_type{VK_OBJECT_TYPE_UNKNOWN};
    std::uint64_t raw{};
};
class StateLocks {
    using Lock = std::variant<std::unique_lock<std::shared_mutex>, std::shared_lock<std::shared_mutex>>;
    std::vector<Lock> locks_;
  public:
    explicit StateLocks(std::span<StateLockRef> states) {
        std::vector<std::shared_mutex*> association_mutexes;
        association_mutexes.reserve(states.size());
        for (const auto& state : states) if (state.association) association_mutexes.push_back(state.association.mutex);
        std::ranges::sort(association_mutexes);
        auto association_end = std::ranges::unique(association_mutexes).begin();
        std::vector<std::shared_lock<std::shared_mutex>> association_locks;
        association_locks.reserve(static_cast<std::size_t>(association_end - association_mutexes.begin()));
        for (auto current = association_mutexes.begin(); current != association_end; ++current) association_locks.emplace_back(**current);
        for (auto& state : states) if (state.association) {
            std::uint64_t stored{};
            state.association.dispatch->vkGetPrivateData(state.association.device, state.object_type, state.raw, state.association.slot, &stored);
            if (!stored) { state.identity = 0; state.mutex = nullptr; continue; }
            auto* lifetime = reinterpret_cast<LifetimeHeader*>(static_cast<std::uintptr_t>(stored));
            state.identity = reinterpret_cast<std::uintptr_t>(lifetime);
            state.mutex = &lifetime->externsync;
        }
        std::ranges::sort(states, {}, [](const StateLockRef& value) { return std::pair(value.identity, !value.exclusive); });
        auto end = std::ranges::unique(states, {}, &StateLockRef::identity).begin();
        for (auto it = states.begin(); it != end; ++it) if (it->mutex) {
            if (it->exclusive) locks_.emplace_back(std::in_place_type<std::unique_lock<std::shared_mutex>>, *it->mutex);
            else locks_.emplace_back(std::in_place_type<std::shared_lock<std::shared_mutex>>, *it->mutex);
        }
    }
};
struct ExternsyncAccess {
    template <typename Handle> static Result<StateLockRef> get(const Handle& value, bool exclusive) {
        if (!value) return StateLockRef{};
        if (!value.ctrl_) return StateLockRef{};
        return StateLockRef{value.id(), &value.ctrl_->externsync, exclusive};
    }
    template <typename Handle> static Result<void> collect(const Handle& value, bool exclusive, std::vector<StateLockRef>& output) {
        auto own = get(value, exclusive);
        if (!own) return std::unexpected(own.error());
        if (own->mutex) output.push_back(*own);
        return {};
    }
    static Result<void> collect(DeviceAssociation association, VkObjectType type, std::uint64_t raw, bool exclusive, std::vector<StateLockRef>& output) {
        if (!association || type == VK_OBJECT_TYPE_UNKNOWN || raw == 0) return {};
        output.push_back(StateLockRef{static_cast<std::uintptr_t>(raw), association.mutex, exclusive, association, type, raw});
        return {};
    }
};
template <typename Handle> [[nodiscard]] bool same_object(const Handle& left, const Handle& right) noexcept {
    if (!left.sameNativeHandle(right)) return false;
    if (left.ctrl_ && right.ctrl_) return left.ctrl_ == right.ctrl_;
    if constexpr (requires { left.parent(); }) return same_object(left.parent(), right.parent());
    return true;
}
} // namespace detail
using DestructionErrorSink = detail::ErrorSink;
inline void setDestructionErrorSink(DestructionErrorSink sink) noexcept { detail::set_error_sink(sink); }
[[nodiscard]] inline DestructionErrorSink destructionErrorSink() noexcept { return detail::error_sink.load(std::memory_order_acquire); }"""

PRELUDE = r"""template <typename Bit, typename Mask>
class Flags {
    Mask mask_{};
  public:
    constexpr Flags() noexcept = default;
    constexpr Flags(Bit bit) noexcept : mask_(static_cast<Mask>(bit)) {}
    constexpr Flags(Mask mask) noexcept : mask_(mask) {}
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return mask_ != Mask{}; }
    [[nodiscard]] constexpr Mask raw() const noexcept { return mask_; }
    [[nodiscard]] constexpr Flags operator|(Flags rhs) const noexcept { return Flags(static_cast<Mask>(mask_ | rhs.mask_)); }
    [[nodiscard]] constexpr Flags operator|(Bit rhs) const noexcept { return *this | Flags(rhs); }
    [[nodiscard]] constexpr Flags operator&(Flags rhs) const noexcept { return Flags(static_cast<Mask>(mask_ & rhs.mask_)); }
    [[nodiscard]] constexpr Flags operator&(Bit rhs) const noexcept { return *this & Flags(rhs); }
    [[nodiscard]] constexpr Flags operator^(Flags rhs) const noexcept { return Flags(static_cast<Mask>(mask_ ^ rhs.mask_)); }
    [[nodiscard]] constexpr Flags operator^(Bit rhs) const noexcept { return *this ^ Flags(rhs); }
    [[nodiscard]] constexpr Flags operator~() const noexcept { return Flags(static_cast<Mask>(~mask_)); }
    constexpr Flags& operator|=(Flags rhs) noexcept { mask_ = static_cast<Mask>(mask_ | rhs.mask_); return *this; }
    constexpr Flags& operator|=(Bit rhs) noexcept { return *this |= Flags(rhs); }
    constexpr Flags& operator&=(Flags rhs) noexcept { mask_ = static_cast<Mask>(mask_ & rhs.mask_); return *this; }
    constexpr Flags& operator&=(Bit rhs) noexcept { return *this &= Flags(rhs); }
    constexpr Flags& operator^=(Flags rhs) noexcept { mask_ = static_cast<Mask>(mask_ ^ rhs.mask_); return *this; }
    constexpr Flags& operator^=(Bit rhs) noexcept { return *this ^= Flags(rhs); }
    [[nodiscard]] constexpr bool operator==(const Flags&) const noexcept = default;
    [[nodiscard]] constexpr bool test(Bit bit) const noexcept { return (mask_ & static_cast<Mask>(bit)) != Mask{}; }
    friend constexpr Flags operator|(Bit lhs, Bit rhs) noexcept { return Flags(lhs) | rhs; }
};

template <typename T> using Result = std::expected<T, ResultCode>;
template <typename T> struct ResultValue { ResultCode status{}; T value{}; };
template <typename Base, typename Extension> struct StructureExtends : std::false_type {};
class ExtensionChain {
    struct Value {
        virtual ~Value() = default;
        virtual std::unique_ptr<Value> clone() const = 0;
        virtual const void* native(const void* next) const = 0;
        virtual void refresh() = 0;
        [[nodiscard]] virtual std::type_index type() const noexcept = 0;
        [[nodiscard]] virtual void* object() noexcept = 0;
        [[nodiscard]] virtual const void* object() const noexcept = 0;
    };
    template <typename T> struct Model final : Value {
        T value;
        mutable typename T::CStruct cache{};
        explicit Model(T input) : value(std::move(input)) {}
        std::unique_ptr<Value> clone() const override { return std::make_unique<Model>(value); }
        const void* native(const void* next) const override {
            value.to_cstruct(&cache);
            auto* tail = reinterpret_cast<VkBaseOutStructure*>(&cache.value);
            while (tail->pNext) tail = tail->pNext;
            tail->pNext = reinterpret_cast<VkBaseOutStructure*>(const_cast<void*>(next));
            return &cache.value;
        }
        void refresh() override { value.from_output_cstruct(cache.value); }
        [[nodiscard]] std::type_index type() const noexcept override { return typeid(T); }
        [[nodiscard]] void* object() noexcept override { return &value; }
        [[nodiscard]] const void* object() const noexcept override { return &value; }
    };
    std::vector<std::unique_ptr<Value>> values_;
  public:
    ExtensionChain() = default;
    ExtensionChain(const ExtensionChain& rhs) { values_.reserve(rhs.values_.size()); for (const auto& value : rhs.values_) values_.push_back(value->clone()); }
    ExtensionChain(ExtensionChain&&) noexcept = default;
    ExtensionChain& operator=(const ExtensionChain& rhs) { ExtensionChain copy(rhs); values_.swap(copy.values_); return *this; }
    ExtensionChain& operator=(ExtensionChain&&) noexcept = default;
    template <typename T> void set(T&& value) { values_.push_back(std::make_unique<Model<std::remove_cvref_t<T>>>(std::forward<T>(value))); }
    void refresh() { for (auto& value : values_) value->refresh(); }
    template <typename T> [[nodiscard]] T* get() noexcept {
        for (auto& value : values_) if (value->type() == typeid(T)) return static_cast<T*>(value->object());
        return nullptr;
    }
    template <typename T> [[nodiscard]] const T* get() const noexcept {
        for (const auto& value : values_) if (value->type() == typeid(T)) return static_cast<const T*>(value->object());
        return nullptr;
    }
    [[nodiscard]] const void* native() const {
        const void* next{};
        for (auto it = values_.rbegin(); it != values_.rend(); ++it) next = (*it)->native(next);
        return next;
    }
};"""


def _vma_sections(vma: VmaModel | None) -> tuple[str, str]:
    if not vma:
        return "", ""
    functions = set(vma.functions)
    allocator_ownership = {"vmaCreateAllocator", "vmaDestroyAllocator"} <= functions
    allocation_ownership = {"vmaAllocateMemory", "vmaFreeMemory"} <= functions
    buffer_ownership = {"vmaCreateBuffer", "vmaDestroyBuffer"} <= functions
    image_ownership = {"vmaCreateImage", "vmaDestroyImage"} <= functions
    declarations: list[str] = []
    implementations: list[str] = []

    view = [
        r"""class AllocationView {
    VmaAllocator allocator_{};
    VmaAllocation allocation_{};
  public:
    AllocationView() noexcept = default;
    AllocationView(VmaAllocator allocator, VmaAllocation allocation) noexcept : allocator_(allocator), allocation_(allocation) {}
    [[nodiscard]] VmaAllocation raw() const noexcept { return allocation_; }
    [[nodiscard]] explicit operator bool() const noexcept { return allocation_ != VmaAllocation{}; }"""
    ]
    if "vmaGetAllocationInfo" in functions:
        view.append("    [[nodiscard]] VmaAllocationInfo information() const noexcept;")
        implementations.append(
            "inline VmaAllocationInfo AllocationView::information() const noexcept { VmaAllocationInfo value{}; if (allocation_) vmaGetAllocationInfo(allocator_, allocation_, &value); return value; }"
        )
    if "vmaMapMemory" in functions:
        view.append("    [[nodiscard]] Result<void*> map() const;")
        implementations.append(
            "inline Result<void*> AllocationView::map() const { void* value{}; auto result = vmaMapMemory(allocator_, allocation_, &value); if (result != VK_SUCCESS) return std::unexpected(static_cast<ResultCode>(result)); return value; }"
        )
    if "vmaUnmapMemory" in functions:
        view.append("    void unmap() const noexcept;")
        implementations.append(
            "inline void AllocationView::unmap() const noexcept { if (allocation_) vmaUnmapMemory(allocator_, allocation_); }"
        )
    if "vmaFlushAllocation" in functions:
        view.append(
            "    [[nodiscard]] Result<void> flush(DeviceSize offset = 0, DeviceSize size = VK_WHOLE_SIZE) const;"
        )
        implementations.append(
            "inline Result<void> AllocationView::flush(DeviceSize offset, DeviceSize size) const { auto result = vmaFlushAllocation(allocator_, allocation_, offset, size); if (result != VK_SUCCESS) return std::unexpected(static_cast<ResultCode>(result)); return {}; }"
        )
    if "vmaInvalidateAllocation" in functions:
        view.append(
            "    [[nodiscard]] Result<void> invalidate(DeviceSize offset = 0, DeviceSize size = VK_WHOLE_SIZE) const;"
        )
        implementations.append(
            "inline Result<void> AllocationView::invalidate(DeviceSize offset, DeviceSize size) const { auto result = vmaInvalidateAllocation(allocator_, allocation_, offset, size); if (result != VK_SUCCESS) return std::unexpected(static_cast<ResultCode>(result)); return {}; }"
        )
    view.append("};")
    declarations.append("\n".join(view))

    if allocation_ownership:
        declarations.append(r"""class Allocation {
    struct State {
        std::shared_ptr<void> allocator_lifetime;
        VmaAllocator allocator{};
        VmaAllocation allocation{};
        State(std::shared_ptr<void> lifetime, VmaAllocator a, VmaAllocation v) : allocator_lifetime(std::move(lifetime)), allocator(a), allocation(v) {}
        ~State();
    };
    std::shared_ptr<State> state_;
    explicit Allocation(std::shared_ptr<State> state) : state_(std::move(state)) {}
    friend class Allocator;
  public:
    Allocation() noexcept = default;
    [[nodiscard]] VmaAllocation raw() const noexcept { return state_ ? state_->allocation : VmaAllocation{}; }
    [[nodiscard]] AllocationView view() const noexcept { return state_ ? AllocationView(state_->allocator, state_->allocation) : AllocationView{}; }
    [[nodiscard]] long use_count() const noexcept { return state_.use_count(); }
    void reset() noexcept { state_.reset(); }
};""")
        implementations.append(
            "inline Allocation::State::~State() { if (allocation) vmaFreeMemory(allocator, allocation); }"
        )

    allocator = ["class Allocator {"]
    if allocator_ownership:
        allocator.append(r"""    struct State {
        VmaAllocator allocator{};
        Device device{};
        State(VmaAllocator a, Device d) : allocator(a), device(std::move(d)) {}
        ~State();
    };
    std::shared_ptr<State> state_;""")
    allocator.extend([r"""    VmaAllocator borrowed_{};
    Device borrowed_device_{};"""])
    if allocator_ownership:
        allocator.append(
            "    explicit Allocator(std::shared_ptr<State> state) : state_(std::move(state)) {}"
        )
    allocator.extend(
        [
            r"""    Allocator(VmaAllocator allocator, const Device& device) : borrowed_(allocator), borrowed_device_(device) {}
  public:
    Allocator() noexcept = default;"""
        ]
    )
    if allocator_ownership:
        implementations.append(
            "inline Allocator::State::~State() { if (allocator) vmaDestroyAllocator(allocator); }"
        )
        allocator.extend(
            [
                "    [[nodiscard]] VmaAllocator raw() const noexcept { return state_ ? state_->allocator : borrowed_; }",
                "    [[nodiscard]] Device device() const noexcept { return state_ ? state_->device : borrowed_device_; }",
                "    [[nodiscard]] long use_count() const noexcept { return state_.use_count(); }",
                "    void reset() noexcept { state_.reset(); borrowed_ = {}; borrowed_device_.reset(); }",
                "    [[nodiscard]] std::shared_ptr<void> lifetime() const noexcept { return state_; }",
                "    [[nodiscard]] static Result<Allocator> create(const Device& device, VmaAllocatorCreateInfo createInfo);",
            ]
        )
        implementations.append(
            "inline Result<Allocator> Allocator::create(const Device& device, VmaAllocatorCreateInfo createInfo) { createInfo.device = device.raw(); VmaAllocator value{}; auto result = vmaCreateAllocator(&createInfo, &value); if (result != VK_SUCCESS) return std::unexpected(static_cast<ResultCode>(result)); try { return Allocator(std::make_shared<State>(value, device)); } catch (...) { vmaDestroyAllocator(value); return std::unexpected(ResultCode::ErrorOutOfHostMemory); } }"
        )
    else:
        allocator.extend(
            [
                "    [[nodiscard]] VmaAllocator raw() const noexcept { return borrowed_; }",
                "    [[nodiscard]] const Device& device() const noexcept { return borrowed_device_; }",
                "    [[nodiscard]] long use_count() const noexcept { return 0; }",
                "    void reset() noexcept { borrowed_ = {}; borrowed_device_.reset(); }",
                "    [[nodiscard]] std::shared_ptr<void> lifetime() const noexcept { return {}; }",
            ]
        )
    allocator.append(
        "    [[nodiscard]] static Allocator borrow(VmaAllocator allocator, const Device& device) { return Allocator(allocator, device); }"
    )
    if allocation_ownership:
        allocator.append(
            "    [[nodiscard]] Result<Allocation> allocate(const MemoryRequirements& requirements, const VmaAllocationCreateInfo& createInfo) const;"
        )
        implementations.append(
            "inline Result<Allocation> Allocator::allocate(const MemoryRequirements& requirements, const VmaAllocationCreateInfo& createInfo) const { MemoryRequirements::CStruct requirementsNative{}; requirements.to_cstruct(&requirementsNative); VmaAllocation value{}; auto result = vmaAllocateMemory(raw(), &requirementsNative.value, &createInfo, &value, nullptr); if (result != VK_SUCCESS) return std::unexpected(static_cast<ResultCode>(result)); try { return Allocation(std::make_shared<Allocation::State>(lifetime(), raw(), value)); } catch (...) { vmaFreeMemory(raw(), value); return std::unexpected(ResultCode::ErrorOutOfHostMemory); } }"
        )
    if buffer_ownership:
        allocator.append(
            "    [[nodiscard]] Result<Buffer> createBuffer(const BufferCreateInfo& bufferInfo, const VmaAllocationCreateInfo& allocationInfo) const;"
        )
        implementations.append(
            "inline Result<Buffer> Allocator::createBuffer(const BufferCreateInfo& bufferInfo, const VmaAllocationCreateInfo& allocationInfo) const { BufferCreateInfo::CStruct bufferNative{}; bufferInfo.to_cstruct(&bufferNative); VkBuffer buffer{}; VmaAllocation allocation{}; VmaAllocationInfo info{}; auto result = vmaCreateBuffer(raw(), &bufferNative.value, &allocationInfo, &buffer, &allocation, &info); if (result != VK_SUCCESS) return std::unexpected(static_cast<ResultCode>(result)); try { return Buffer::adoptVma(buffer, device(), lifetime(), raw(), allocation, info, allocationInfo, std::make_shared<const BufferCreateInfo>(bufferInfo)); } catch (...) { vmaDestroyBuffer(raw(), buffer, allocation); return std::unexpected(ResultCode::ErrorOutOfHostMemory); } }"
        )
    if image_ownership:
        allocator.append(
            "    [[nodiscard]] Result<Image> createImage(const ImageCreateInfo& imageInfo, const VmaAllocationCreateInfo& allocationInfo) const;"
        )
        implementations.append(
            "inline Result<Image> Allocator::createImage(const ImageCreateInfo& imageInfo, const VmaAllocationCreateInfo& allocationInfo) const { ImageCreateInfo::CStruct imageNative{}; imageInfo.to_cstruct(&imageNative); VkImage image{}; VmaAllocation allocation{}; VmaAllocationInfo info{}; auto result = vmaCreateImage(raw(), &imageNative.value, &allocationInfo, &image, &allocation, &info); if (result != VK_SUCCESS) return std::unexpected(static_cast<ResultCode>(result)); try { return Image::adoptVma(image, device(), lifetime(), raw(), allocation, info, allocationInfo, std::make_shared<const ImageCreateInfo>(imageInfo)); } catch (...) { vmaDestroyImage(raw(), image, allocation); return std::unexpected(ResultCode::ErrorOutOfHostMemory); } }"
        )
    allocator.append("};")
    declarations.append("\n".join(allocator))
    metadata = [
        f"// parsed VMA function: {fn.return_type} {fn.name}({', '.join(p.type for p in fn.parameters)})"
        for fn in vma.functions.values()
    ]
    return "\n".join(declarations), "\n\n".join([*implementations, *metadata])


def _vma_resource_types(vma: VmaModel | None) -> frozenset[str]:
    if not vma:
        return frozenset()
    functions = set(vma.functions)
    resources: set[str] = set()
    if {"vmaCreateBuffer", "vmaDestroyBuffer"} <= functions:
        resources.add("VkBuffer")
    if {"vmaCreateImage", "vmaDestroyImage"} <= functions:
        resources.add("VkImage")
    return frozenset(resources)


def emit_sections(
    registry: Registry,
    config: GeneratorConfig,
    template: Template,
    vma: VmaModel | None = None,
) -> dict[str, str]:
    analysis = analyze(registry, config)
    known = {_cpp_type(item.name, registry, config) for item in registry.types.values()}
    unknown_injections = set(template.injections) - known
    if unknown_injections:
        # render_template also checks this; fail before mutating IR with injections.
        from .template import TemplateError

        raise TemplateError(
            f"injections target unknown types: {', '.join(sorted(unknown_injections))}"
        )
    vma_decl, vma_impl = _vma_sections(vma)
    vma_resources = _vma_resource_types(vma)
    struct_impl = _emit_struct_implementations(registry, config)
    handle_impl = _emit_handle_implementations(
        analysis, registry, config, vma_resources
    )
    # A template containing both declarations and definitions is a
    # header-only/module-style output and needs ODR-safe inline definitions.
    # A definition-only template (the paired .cpp mode) emits ordinary
    # external definitions instead.
    combined_output = (
        "{{handles}}" in template.text and "{{handle_implementations}}" in template.text
    )
    if not combined_output:
        struct_impl = re.sub(r"(?m)^inline ", "", struct_impl)
        handle_impl = re.sub(r"(?m)^inline ", "", handle_impl)
        vma_impl = re.sub(r"(?m)^inline ", "", vma_impl)
    return {
        "generated_notice": "// Generated by vulkan-wrapper-gen. Do not edit.\n",
        "namespace": config.namespace,
        "module_name": config.module,
        "includes": "#include <volk.h>\n"
        + (f"#include <{config.vma_include}>\n" if vma else "")
        + "#include <algorithm>\n#include <array>\n#include <atomic>\n#include <cstdint>\n#include <cstring>\n#include <expected>\n#include <functional>\n#include <limits>\n#include <memory>\n#include <mutex>\n#include <new>\n#include <optional>\n#include <ranges>\n#include <shared_mutex>\n#include <span>\n#include <string>\n#include <string_view>\n#include <typeindex>\n#include <type_traits>\n#include <unordered_map>\n#include <utility>\n#include <variant>\n#include <vector>",
        "forward_declarations": _emit_forwards(registry, config)
        + "\n"
        + _emit_command_result_forwards(analysis, registry),
        "result_code": _emit_result_code(registry),
        "aliases": _emit_aliases(registry, config),
        "constants": _emit_constants(registry),
        "enums": _emit_enums(registry, config),
        "runtime_declarations": PRELUDE + "\n" + RUNTIME,
        "runtime_implementations": "",
        "structure_extensions": _emit_extensions(registry, config),
        "structs": _emit_structs(registry, config, template),
        "handles": _emit_handles(analysis, registry, config, template, vma_resources),
        "context": "",
        "command_declarations": _emit_command_metadata(analysis, registry, config),
        "command_implementations": "",
        "struct_implementations": struct_impl,
        "struct_template_implementations": "",
        "handle_implementations": handle_impl,
        "handle_template_implementations": _emit_handle_template_implementations(
            analysis, registry, config
        ),
        "command_template_implementations": "",
        "vma_declarations": vma_decl,
        "vma_implementations": vma_impl,
    }
