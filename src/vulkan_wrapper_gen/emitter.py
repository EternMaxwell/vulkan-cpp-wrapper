"""C++23 wrapper emitter driven directly by the middle-layer IR.

The emitter consumes :class:`vulkan_wrapper_gen.ir.IrRegistry` and nothing
else from the XML pipeline.  The IR already carries the derived facts the
previous analysis layer re-derived from the raw registry (receivers, member
names, output shapes, creation records, releasers, array/count links), so the
emitter reads them directly rather than reconstructing them.
"""

from __future__ import annotations

import re

from .config import GeneratorConfig
from .ir.model import (
    Command,
    Enum,
    EnumValue,
    FuncPointer,
    Handle,
    IrRegistry,
    Param,
    Struct,
)
from .naming import constant_name, enum_name
from .template import Template
from .vma import VmaModel


# ---------------------------------------------------------------------------
# Name/type helpers
# ---------------------------------------------------------------------------

def _guard(text: str, protect: str | None) -> str:
    return (
        f"#if defined({protect})\n{text}\n#endif // defined({protect})"
        if protect
        else text
    )


def _doc_comment(doc: str | None, config: GeneratorConfig, indent: str = "") -> str:
    """Render an optional doc string as a Doxygen ``///`` comment (or '').

    Only emitted when ``config.emit_docs`` is enabled; whitespace is collapsed
    so a multi-part doc reads as a single line.
    """
    if not config.emit_docs or not doc:
        return ""
    text = " ".join(doc.split())
    return f"{indent}/// {text}\n"


def _c_name(ir: IrRegistry, general: str) -> str | None:
    """General name -> exact C spelling by consulting every IR collection."""
    for collection in (
        ir.handles,
        ir.structs,
        ir.enums,
        ir.bitmasks,
        ir.basetypes,
        ir.func_pointers,
        ir.aliases,
        ir.defines,
        ir.raw_types,
    ):
        item = collection.get(general)
        if item is not None:
            return item.c_name
    return None


def _cpp_type(name: str, ir: IrRegistry, config: GeneratorConfig) -> str:
    """Map an IR general type name to its C++ wrapper spelling."""
    if name == "Result":
        return "ResultCode"
    if name in {"Flags", "Flags64"}:
        return "Vk" + name
    c_name = _c_name(ir, name)
    if c_name and c_name in config.type_names:
        return config.type_names[c_name]
    alias = ir.aliases.get(name)
    if alias is not None:
        return _cpp_type(alias.target, ir, config)
    if name in ir.enums and ir.enums[name].kind in {"enum", "bitmask"}:
        return name
    if name in ir.handles or name in ir.structs or name in ir.bitmasks:
        return name
    if name in ir.basetypes or name in ir.func_pointers or name in ir.raw_types:
        return name
    return name


def _type_category(name: str, ir: IrRegistry) -> str | None:
    """Resolved category of an IR general type name.

    Mirrors the old registry ordering: explicit ``<type>`` declarations win
    over enum groups, so a ``FlagBits`` enum declared with
    ``<type category="enum">`` is an enum, not a bitmask typedef.
    """
    if name in ir.handles:
        return "handle"
    if name in ir.structs:
        return ir.structs[name].category
    if name in ir.bitmasks:
        return "bitmask"
    if name in ir.basetypes:
        return "basetype"
    if name in ir.func_pointers:
        return "funcpointer"
    alias = ir.aliases.get(name)
    if alias is not None:
        return alias.resolved_category or _type_category(alias.target, ir)
    if name in ir.raw_types:
        return ir.raw_types[name].category
    if name in ir.enums:
        return ir.enums[name].kind if ir.enums[name].kind in {"enum", "bitmask"} else None
    return None


def _as_struct(ir: IrRegistry, name: str) -> Struct | None:
    resolved = ir.resolve(name)
    return resolved if isinstance(resolved, Struct) else None


def _as_handle(ir: IrRegistry, name: str) -> Handle | None:
    resolved = ir.resolve(name)
    return resolved if isinstance(resolved, Handle) else None


def _is_opaque_raw(name: str, ir: IrRegistry) -> bool:
    """True for unmodeled foreign/platform tokens.

    ``SECURITY_ATTRIBUTES``, ``Display``, ``HWND`` and friends carry no XML
    category (the raw record's ``category`` is ``None``) and are only
    forward-declared by volk, so they cannot be wrapped by value; they must
    pass through as raw pointers.  Enum/bitmask typedefs (``VkResult``,
    ``VkFormat``, ...) also live in ``raw_types`` but have a real category and
    a modeled wrapper, so they are excluded here.
    """
    raw = ir.raw_types.get(name)
    return raw is not None and raw.category is None


def _lengths(param: Param) -> tuple[str, ...]:
    return tuple(length.text for length in param.lengths)


def _native_type(param: Param) -> str:
    prefix = "const " if param.const else ""
    # Array parameters in C declarations decay to pointers even though the XML
    # spelling has no `*` token (for example `float values[4]`).
    depth = param.pointer_depth + (1 if "[" in param.c_suffix else 0)
    return prefix + param.c_type + "*" * depth


def _public_param_name(param: Param) -> str:
    return param.public_name


def _public_param_type(
    param: Param, ir: IrRegistry, config: GeneratorConfig
) -> str:
    cpp = _cpp_type(param.type, ir, config)
    category = _type_category(param.type, ir)
    array_sizes = re.findall(r"\[([^\]]+)\]", param.c_suffix)
    if array_sizes:
        extent = ", ".join(array_sizes)
        return f"std::span<{'const ' if param.const else ''}{cpp}, {extent}>"
    if param.pointer_depth == 0 and not param.c_suffix:
        # Handle wrappers already carry an empty/null state, so an optional
        # handle is just a plain (possibly default-constructed) handle ref.
        if category == "handle":
            return f"const {cpp}&"
        return cpp
    if (
        param.const
        and param.type == "char"
        and param.pointer_depth == 1
        and "null-terminated" in _lengths(param)
    ):
        return (
            "std::optional<std::string_view>"
            if param.is_optional
            else "std::string_view"
        )
    lengths = _lengths(param)
    if lengths and "null-terminated" not in lengths:
        if param.type == "void":
            return (
                "std::span<const std::byte>" if param.const else "std::span<std::byte>"
            )
        return f"std::span<{'const ' if param.const else ''}{cpp}>"
    is_output = param.direction == "output"
    if is_output:
        if param.type == "void":
            return "void" + "*" * param.pointer_depth
        return f"{cpp}*"
    if param.pointer_depth == 1 and category in {"struct", "union"}:
        if param.is_optional:
            return f"std::optional<std::reference_wrapper<const {cpp}>>"
        return f"const {cpp}&"
    if category is not None:
        prefix = "const " if param.const else ""
        return prefix + cpp + "*" * param.pointer_depth
    return _native_type(param)


def _public_argument(
    param: Param, ir: IrRegistry, name: str | None = None
) -> str:
    name = name or _public_param_name(param)
    lengths = _lengths(param)
    if (lengths and "null-terminated" not in lengths) or "[" in param.c_suffix:
        if param.type == "void":
            return f"reinterpret_cast<{'const ' if param.const else ''}void*>({name}.empty() ? nullptr : {name}.data())"
        return f"{name}.empty() ? nullptr : {name}.data()"
    struct = _as_struct(ir, param.type)
    if (
        param.const
        and param.pointer_depth == 1
        and struct is not None
        and struct.category in {"struct", "union"}
    ):
        if param.is_optional:
            return f"{name} ? &{name}->get() : nullptr"
        return f"&{name}"
    if param.pointer_depth == 0:
        category = _type_category(param.type, ir)
        if category == "handle":
            return f"{name}.raw()"
        if category in {"enum", "bitmask"}:
            return _native_value(param.type, name, ir)
    return name


def _needs_native_conversion(param: Param, ir: IrRegistry) -> bool:
    lengths = _lengths(param)
    if (lengths and "null-terminated" not in lengths) or "[" in param.c_suffix:
        return True
    category = _type_category(param.type, ir)
    if category is None:
        return False
    return category in {"struct", "union"} or (
        (bool(lengths) or param.pointer_depth > 0 or "[" in param.c_suffix)
        and category in {"enum", "bitmask", "handle"}
    )


def _enum_value(value: EnumValue, group: Enum) -> str:
    if value.alias_of:
        return value.alias_of
    if value.value is not None:
        return value.value
    if value.bitpos is not None:
        suffix = "ULL" if (group.bitwidth or 32) > 32 else "U"
        return f"(1{suffix} << {value.bitpos})"
    if value.offset is not None and value.extnumber is not None:
        number = 1_000_000_000 + (value.extnumber - 1) * 1_000 + value.offset
        return str(-number if value.negative else number)
    return value.name


def _is_typed_bitmask(name: str, ir: IrRegistry) -> bool:
    alias = ir.aliases.get(name)
    if alias is not None:
        return _is_typed_bitmask(alias.target, ir)
    bitmask = ir.bitmasks.get(name)
    if bitmask is None or not bitmask.bits:
        return False
    group = ir.enums.get(bitmask.bits)
    return bool(group and group.values and bitmask.bits in ir.enums)


def _native_value(name: str, expression: str, ir: IrRegistry) -> str:
    if _type_category(name, ir) == "bitmask" and _is_typed_bitmask(name, ir):
        return f"{expression}.raw()"
    return f"static_cast<{_c_name(ir, name) or name}>({expression})"


def _has_native_definition(struct: Struct) -> bool:
    return bool(struct.members)


def _has_pnext(name: str, ir: IrRegistry) -> bool:
    struct = _as_struct(ir, name)
    return bool(struct and any(member.name == "pNext" for member in struct.members))


def _output_chain_refresh(name: str, expression: str, ir: IrRegistry) -> str:
    return (
        f" {expression}.nextInChain.refresh();"
        if _has_pnext(name, ir)
        else ""
    )


def _native_type_name(name: str, ir: IrRegistry) -> str:
    """Name a C type without resolving to a same-named wrapper."""
    struct = _as_struct(ir, name)
    c_name = _c_name(ir, name) or name
    if struct and struct.members and not c_name.startswith("Vk"):
        return f"::{c_name}"
    return c_name


# ---------------------------------------------------------------------------
# Enums / aliases / constants
# ---------------------------------------------------------------------------

def _emit_enums(ir: IrRegistry, config: GeneratorConfig) -> str:
    output: list[str] = []
    for group in ir.enums.values():
        if group.kind not in {"enum", "bitmask"} or not group.values:
            continue
        if group.c_name == "VkResult" or not group.c_name.startswith("Vk"):
            continue
        cpp = _cpp_type(group.name, ir, config)
        underlying = "std::uint64_t" if group.bitwidth == 64 else "std::int32_t"
        values: list[str] = []
        used: set[str] = set()
        for value in group.values:
            if not value.active:
                continue
            name = enum_name(group.c_name, value.name, ir.tags)
            if name in used:
                name += "_" + value.name.rsplit("_", 1)[-1]
            used.add(name)
            values.append(
                _doc_comment(value.doc, config, "    ")
                + _guard(
                    f"    {name} = static_cast<{underlying}>({_enum_value(value, group)}),",
                    value.protect,
                )
            )
        output.append(
            _doc_comment(group.doc or group.availability.doc, config)
            + _guard(
                f"enum class {cpp} : {underlying} {{\n" + "\n".join(values) + "\n};",
                group.availability.protect,
            )
        )
    return "\n\n".join(output)


def _emit_aliases(ir: IrRegistry, config: GeneratorConfig) -> str:
    result: list[str] = []
    for general in ir.type_order:
        alias = ir.aliases.get(general)
        if alias is not None:
            if not alias.active:
                continue
            alias_cpp = config.type_names.get(alias.c_name, alias.name)
            target_c = _c_name(ir, alias.target)
            if target_c is None:
                continue
            target_group = ir.enums.get(alias.target)
            if target_group is not None and not target_group.values:
                continue
            target_cpp = _cpp_type(alias.target, ir, config)
            if alias_cpp != target_cpp:
                result.append(
                    _guard(
                        f"using {alias_cpp} = {target_cpp};",
                        alias.protect or alias.availability.protect,
                    )
                )
            continue
        basetype = ir.basetypes.get(general)
        if basetype is not None:
            if not basetype.active or basetype.c_name in {"VkFlags", "VkFlags64"}:
                continue
            result.append(
                _guard(
                    f"using {basetype.name} = {basetype.c_name};",
                    basetype.protect or basetype.availability.protect,
                )
            )
            continue
        func_pointer = ir.func_pointers.get(general)
        if func_pointer is not None:
            if not func_pointer.active:
                continue
            result.append(
                _guard(
                    f"using {func_pointer.name} = {func_pointer.c_name};",
                    func_pointer.protect or func_pointer.availability.protect,
                )
            )
            continue
        bitmask = ir.bitmasks.get(general)
        if bitmask is not None:
            if not bitmask.active or bitmask.c_name in {"VkFlags", "VkFlags64"}:
                continue
            if _is_typed_bitmask(bitmask.name, ir):
                bits_cpp = _cpp_type(bitmask.bits, ir, config)
                result.append(
                    _guard(
                        f"template <> struct FlagTraits<{bits_cpp}> {{ using MaskType = {bitmask.c_name}; }};",
                        bitmask.protect or bitmask.availability.protect,
                    )
                )
                result.append(
                    _guard(
                        f"using {bitmask.name} = Flags<{bits_cpp}, {bitmask.c_name}>;",
                        bitmask.protect or bitmask.availability.protect,
                    )
                )
            else:
                result.append(
                    _guard(
                        f"using {bitmask.name} = {bitmask.c_name};",
                        bitmask.protect or bitmask.availability.protect,
                    )
                )
            continue
        struct = _as_struct(ir, general)
        if (
            struct is not None
            and struct.category == "union"
            and struct.members
            and struct.active
        ):
            result.append(
                _guard(
                    f"using {_cpp_type(struct.name, ir, config)} = {struct.c_name};",
                    struct.protect or struct.availability.protect,
                )
            )
    return "\n".join(result)


def _emit_constants(ir: IrRegistry, config: GeneratorConfig) -> str:
    lines: list[str] = []
    emitted: set[str] = set()
    for item in ir.constants.values():
        if not item.active:
            continue
        name = constant_name(item.c_name, ir.tags)
        if name in emitted:
            continue
        emitted.add(name)
        if item.alias_of:
            target = constant_name(item.alias_of, ir.tags)
            declaration = f"inline constexpr auto {name} = {target};"
        elif item.value is not None:
            native_type = item.c_type or "auto"
            declaration = (
                f"inline constexpr auto {name} = {item.c_name};"
                if native_type == "auto"
                else f"inline constexpr {native_type} {name} = static_cast<{native_type}>({item.c_name});"
            )
        else:
            continue
        lines.append(
            _doc_comment(item.doc or item.availability.doc, config)
            + _guard(declaration, item.protect or item.availability.protect)
        )
    return "\n".join(lines)


def _member_cpp(param: Param, ir: IrRegistry, config: GeneratorConfig) -> str:
    value = _cpp_type(param.type, ir, config)
    array_sizes = re.findall(r"\[([^\]]+)\]", param.c_suffix)
    if array_sizes:
        for size in reversed(array_sizes):
            value = f"std::array<{value}, {size}>"
        return value
    if param.pointer_depth:
        lengths = _lengths(param)
        if (
            param.const
            and param.type == "char"
            and param.pointer_depth == 2
            and lengths
        ):
            return "std::vector<std::string>"
        if value == "void" and lengths:
            return (
                "std::vector<std::byte>"
                if param.pointer_depth == 1
                else "std::vector<const void*>"
            )
        if (
            param.const
            and param.type == "char"
            and "null-terminated" in lengths
        ):
            return "std::string"
        if lengths and any(length != "null-terminated" for length in lengths):
            return f"std::vector<{value}>"
        if (
            param.pointer_depth == 1
            and _type_category(param.type, ir) == "struct"
        ):
            return f"std::optional<{value}>" if param.is_optional else value
        if (
            param.pointer_depth == 1
            and _type_category(param.type, ir) == "native_struct"
        ):
            return f"std::optional<{value}>"
        if param.pointer_depth == 1 and _type_category(param.type, ir) in {
            "enum",
            "bitmask",
        }:
            return f"std::optional<{value}>"
        if param.pointer_depth == 1 and _is_opaque_raw(param.type, ir):
            # Opaque foreign/platform tokens (SECURITY_ATTRIBUTES, Display, ...)
            # are only forward-declared by volk, so they cannot be held by value
            # in a std::optional. Pass the pointer through unchanged.
            return ("const " if param.const else "") + value + "*"
        if param.pointer_depth == 1 and param.is_optional and value != "void":
            return f"std::optional<{value}>"
        if ir.resolve(param.type) is not None:
            return (
                ("const " if param.const else "") + value + "*" * param.pointer_depth
            )
        return _native_type(param)
    return value


def _safe_default(param: Param, cpp_type: str) -> str:
    if param.values:
        return f"{{static_cast<{cpp_type}>({param.values})}}"
    return "{}"


# ---------------------------------------------------------------------------
# Struct helpers
# ---------------------------------------------------------------------------

def _count_sources(struct: Struct) -> dict[str, Param]:
    result: dict[str, Param] = {}
    member_names = {member.name for member in struct.members}
    for member in struct.members:
        if "[" in member.c_suffix:
            continue
        for length in _lengths(member):
            if re.fullmatch(r"[A-Za-z_]\w*", length):
                result.setdefault(length, member)
        expression = (member.alt_length or "").strip()
        quotient = re.fullmatch(r"([A-Za-z_]\w*)\s*/\s*([1-9]\d*)", expression)
        if quotient and quotient.group(1) in member_names:
            result.setdefault(quotient.group(1), member)
    return result


def _array_count_field(struct: Struct, member: Param) -> tuple[Param, int] | None:
    """The scalar count field an array member is sized by, plus the element
    multiplier for that length, so the array setter can also write the count.

    A byte-length count such as ``codeSize / 4`` multiplies the element count
    back out to recover the byte size."""
    names = {m.name: m for m in struct.members}
    expression = (member.alt_length or "").strip()
    quotient = re.fullmatch(r"([A-Za-z_]\w*)\s*/\s*([1-9]\d*)", expression)
    if quotient:
        count = names.get(quotient.group(1))
        if count is not None and count.pointer_depth == 0 and not count.c_suffix:
            return count, int(quotient.group(2))
        return None
    for length in _lengths(member):
        if re.fullmatch(r"[A-Za-z_]\w*", length):
            count = names.get(length)
            if count is not None and count.pointer_depth == 0 and not count.c_suffix:
                return count, 1
    return None


def _context_length_name(key: str) -> str:
    parts = key.split("_")
    return "context" + "".join(part[:1].upper() + part[1:] for part in parts)


def _direct_context_lengths(struct: Struct) -> tuple[str, ...]:
    result: list[str] = []
    for member in struct.members:
        for expression in (
            *_lengths(member),
            *((member.alt_length,) if member.alt_length else ()),
        ):
            for match in re.finditer(r"\*_([A-Za-z_]\w*)", expression):
                if match.group(1) not in result:
                    result.append(match.group(1))
    return tuple(result)


def _context_length_source(struct: Struct, key: str) -> Param | None:
    suffix = "_" + key
    return next(
        (
            member
            for member in struct.members
            if member.pointer_depth == 0
            and (member.name == key or member.name.endswith(suffix))
        ),
        None,
    )


def _struct_context_lengths(
    struct: Struct, ir: IrRegistry, visiting: set[str] | None = None
) -> tuple[str, ...]:
    visiting = set() if visiting is None else visiting
    if struct.name in visiting:
        return ()
    visiting.add(struct.name)
    required = list(_direct_context_lengths(struct))
    for member in struct.members:
        nested = _as_struct(ir, member.type)
        if not nested or nested.category != "struct" or not nested.members:
            continue
        for key in _struct_context_lengths(nested, ir, visiting):
            if _context_length_source(struct, key) is None and key not in required:
                required.append(key)
    visiting.remove(struct.name)
    return tuple(required)


def _native_array_size(param: Param, struct: Struct) -> str | None:
    expression = param.alt_length or next(
        (length for length in _lengths(param) if length != "null-terminated"), None
    )
    if not expression or expression == "1":
        return expression
    expression = re.sub(
        r"\*_([A-Za-z_]\w*)",
        lambda match: _context_length_name(match.group(1)),
        expression,
    )
    names = {value.name for value in struct.members}
    return re.sub(
        r"\b[A-Za-z_]\w*\b",
        lambda match: (
            f"native.{match.group(0)}" if match.group(0) in names else match.group(0)
        ),
        expression,
    )


def _struct_from_parent_types(
    struct: Struct, ir: IrRegistry, visiting: set[str] | None = None
) -> tuple[str, ...]:
    visiting = set() if visiting is None else visiting
    if struct.name in visiting:
        return ()
    visiting.add(struct.name)
    result: list[str] = []
    for member in struct.members:
        category = _type_category(member.type, ir)
        if category == "handle":
            handle = _as_handle(ir, member.type)
            parent = handle.parent if handle else None
            if parent and parent not in result:
                result.append(parent)
        elif category == "struct":
            nested = _as_struct(ir, member.type)
            if nested:
                for parent in _struct_from_parent_types(nested, ir, visiting):
                    if parent not in result:
                        result.append(parent)
    visiting.remove(struct.name)
    return tuple(result)


def _from_parent_name(
    type_name: str, ir: IrRegistry, config: GeneratorConfig
) -> str:
    return "owner" + _cpp_type(type_name, ir, config)


def _struct_from_parameters(
    struct: Struct, ir: IrRegistry, config: GeneratorConfig
) -> str:
    parents = "".join(
        f", const {_cpp_type(parent, ir, config)}& {_from_parent_name(parent, ir, config)}"
        for parent in _struct_from_parent_types(struct, ir)
    )
    contexts = "".join(
        f", std::size_t {_context_length_name(key)}"
        for key in _struct_context_lengths(struct, ir)
    )
    return parents + contexts


def _nested_from_arguments(
    type_name: str,
    ir: IrRegistry,
    config: GeneratorConfig,
    container: Struct | None = None,
) -> str:
    nested = _as_struct(ir, type_name)
    if not nested:
        return ""
    parents = "".join(
        f", {_from_parent_name(parent, ir, config)}"
        for parent in _struct_from_parent_types(nested, ir)
    )
    contexts: list[str] = []
    for key in _struct_context_lengths(nested, ir):
        source = _context_length_source(container, key) if container else None
        expression = f"native.{source.name}" if source else _context_length_name(key)
        contexts.append(f", static_cast<std::size_t>({expression})")
    return parents + "".join(contexts)


def _borrow_handle_lines(
    target: str,
    source: str,
    type_name: str,
    ir: IrRegistry,
    config: GeneratorConfig,
    indent: str = "    ",
) -> list[str]:
    cpp = _cpp_type(type_name, ir, config)
    handle = ir.handles[type_name]
    parent = handle.parent
    parent_arg = f", {_from_parent_name(parent, ir, config)}" if parent else ""
    c_type = handle.c_name
    return [
        f"{indent}if ({source} == {c_type}{{}}) {target}.reset();",
        f'{indent}else {{ auto wrapped = {cpp}::borrow({source}{parent_arg}); if (wrapped) {target} = std::move(*wrapped); else {{ {target}.reset(); detail::report_error(wrapped.error(), "{cpp}", detail::raw_key({source})); }} }}',
    ]


def _struct_member_names(struct: Struct) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for member in struct.members:
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


def _callback_groups(struct: Struct, ir: IrRegistry) -> tuple[tuple[str, tuple[Param, ...]], ...]:
    """Group a struct's callback members by their userdata carrier field.

    A callback is a funcpointer-typed member whose funcpointer type declares a
    ``void*`` parameter named after one of the struct's own ``void*`` fields
    (Vulkan's convention is the ``pUserData`` carrier). Returns one group per
    carrier, keyed by the carrier member name. Funcpointer members without a
    matching carrier (e.g. a plain ``PFN_vkGetInstanceProcAddr`` slot) are not
    grouped and stay ordinary raw fields.
    """
    userdata_names = {
        member.name
        for member in struct.members
        if member.type == "void" and member.pointer_depth == 1
    }
    groups: dict[str, list[Param]] = {}
    for member in struct.members:
        func_pointer = ir.func_pointers.get(member.type)
        if func_pointer is None:
            continue
        for param in func_pointer.params:
            if (
                param.type == "void"
                and param.pointer_depth == 1
                and param.name in userdata_names
            ):
                groups.setdefault(param.name, []).append(member)
                break
    return tuple((name, tuple(members)) for name, members in groups.items())


def _callback_field_name(member: Param) -> str:
    """C++ field name for a callback member (strip the leading ``pfn`` prefix)."""
    name = member.name[3:] if member.name.startswith("pfn") else member.name
    return name[:1].lower() + name[1:]


def _callback_userdata_param(func_pointer: FuncPointer, carrier: str) -> Param | None:
    for param in func_pointer.params:
        if param.type == "void" and param.pointer_depth == 1 and param.name == carrier:
            return param
    return None


def _callback_callable_signature(func_pointer: FuncPointer, carrier: str) -> str:
    """The ``std::function<Ret(Args...)>`` type a callback field stores.

    Args is the funcpointer's native parameter list minus the userdata carrier
    (which the generated trampoline consumes to recover the bundle).
    """
    userdata = _callback_userdata_param(func_pointer, carrier)
    arguments = [
        _native_type(param)
        for param in func_pointer.params
        if param is not userdata
    ]
    return f"std::function<{func_pointer.c_return_type or 'void'}({', '.join(arguments)})>"


def _callback_trampoline_arguments(
    func_pointer: FuncPointer, carrier: str
) -> tuple[str, str]:
    """The non-userdata native argument names and the userdata param name."""
    arguments: list[str] = []
    userdata_name = ""
    for param in func_pointer.params:
        if param.name == carrier:
            userdata_name = param.name
        else:
            arguments.append(param.name)
    return ", ".join(arguments), userdata_name


def _cstruct_cache_lines(
    struct: Struct, ir: IrRegistry, config: GeneratorConfig
) -> list[str]:
    lines: list[str] = []
    for member in struct.members:
        category = _type_category(member.type, ir)
        cpp = _cpp_type(member.type, ir, config)
        array_sizes = re.findall(r"\[([^\]]+)\]", member.c_suffix)
        lengths = _lengths(member)
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
            and lengths
        ):
            lines.append(f"        std::vector<const char*> {member.name}_native;")
        elif not array_sizes and lengths and category == "struct":
            lines.append(f"        std::vector<{cpp}::CStruct> {member.name}_cache;")
            lines.append(
                f"        std::vector<{_native_type_name(member.type, ir)}> {member.name}_native;"
            )
            if member.pointer_depth > 1:
                lines.append(
                    f"        std::vector<const {_native_type_name(member.type, ir)}*> {member.name}_pointers;"
                )
        elif (
            not array_sizes
            and lengths
            and category in {"handle", "enum", "bitmask"}
        ):
            lines.append(
                f"        std::vector<{_native_type_name(member.type, ir)}> {member.name}_native;"
            )
        elif member.pointer_depth == 1 and category == "struct":
            if member.is_optional:
                lines.append(
                    f"        std::optional<{cpp}::CStruct> {member.name}_cache;"
                )
            else:
                lines.append(f"        {cpp}::CStruct {member.name}_cache{{}};")
        elif (
            member.pointer_depth == 1
            and (member.is_optional or category in {"enum", "bitmask"})
            and member.type != "void"
            and not _is_opaque_raw(member.type, ir)
        ):
            lines.append(
                f"        std::optional<{_native_type_name(member.type, ir)}> {member.name}_native;"
            )
        elif (
            member.pointer_depth == 0
            and category == "struct"
            and "[" not in member.c_suffix
        ):
            lines.append(f"        {cpp}::CStruct {member.name}_cache{{}};")
    return lines


def _emit_struct(
    struct: Struct, ir: IrRegistry, config: GeneratorConfig, injection: list[str]
) -> str:
    name = _cpp_type(struct.name, ir, config)
    if not struct.members:
        return _doc_comment(struct.doc or struct.availability.doc, config) + _guard(
            f"using {name} = {struct.c_name};",
            struct.protect or struct.availability.protect,
        )
    if struct.category == "union":
        return _doc_comment(struct.doc or struct.availability.doc, config) + _guard(
            f"using {name} = {struct.c_name};",
            struct.protect or struct.availability.protect,
        )
    field_names = _struct_member_names(struct)
    callback_groups = _callback_groups(struct, ir)
    callback_members = {member.name for _, members in callback_groups for member in members}
    carrier_members = {carrier for carrier, _ in callback_groups}
    lines = [
        f"struct {name} {{",
        f"    using native_type = {_native_type_name(struct.name, ir)};",
        "    static constexpr bool binary_compatible = false;",
        "    struct CStruct {",
        "        native_type value{};",
    ]
    lines.extend(_cstruct_cache_lines(struct, ir, config))
    lines.extend(["    };"])
    for member in struct.members:
        if member.name not in field_names:
            continue
        if member.name in callback_members or member.name in carrier_members:
            continue
        field_name = field_names[member.name]
        doc_prefix = _doc_comment(member.doc, config, "    ")
        if member.name == "sType" and member.values:
            cpp = _cpp_type(member.type, ir, config)
            lines.append(
                doc_prefix + f"    {cpp} {field_name}{{static_cast<{cpp}>({member.values})}};"
            )
        elif member.name == "pNext":
            lines.append(doc_prefix + "    ExtensionChain nextInChain{};")
        else:
            member_type = _member_cpp(member, ir, config)
            lines.append(
                doc_prefix + f"    {member_type} {field_name}{_safe_default(member, member_type)};"
            )
    if callback_groups:
        # A refcounted bundle holds one std::function per callback. Struct
        # copies share the bundle (shared_ptr), so captured state lives as long
        # as any copy of the create-info or allocator, which is exactly the
        # lifetime the native callback may still be invoked on.
        lines.append("    struct Callbacks {")
        for carrier, members in callback_groups:
            for member in members:
                func_pointer = ir.func_pointers[member.type]
                field = _callback_field_name(member)
                signature = _callback_callable_signature(func_pointer, carrier)
                lines.append(f"        {signature} {field}{{}};")
        lines.append("    };")
        lines.append("    std::shared_ptr<Callbacks> callbacks_{};")
    for member in struct.members:
        if member.name in {"sType", "pNext"} or member.name not in field_names:
            continue
        if member.name in callback_members or member.name in carrier_members:
            continue
        field_name = field_names[member.name]
        method = field_name[:1].upper() + field_name[1:]
        cpp = _member_cpp(member, ir, config)
        count = _array_count_field(struct, member)
        count_stmt = ""
        if count is not None:
            count_param, multiplier = count
            count_field = field_names[count_param.name]
            count_type = _member_cpp(count_param, ir, config)
            size_expr = f"{field_name}.size()"
            if multiplier != 1:
                size_expr += f" * {multiplier}"
            count_stmt = (
                f" {count_field} = static_cast<{count_type}>({size_expr});"
            )
        lines.append(
            f"    {name}& set{method}({cpp} value) & {{ {field_name} = std::move(value);{count_stmt} return *this; }}"
        )
        lines.append(
            f"    {name}&& set{method}({cpp} value) && {{ {field_name} = std::move(value);{count_stmt} return std::move(*this); }}"
        )
    for carrier, members in callback_groups:
        for member in members:
            func_pointer = ir.func_pointers[member.type]
            field = _callback_field_name(member)
            signature = _callback_callable_signature(func_pointer, carrier)
            method = field[:1].upper() + field[1:]
            lazy = f"if (!callbacks_) callbacks_ = std::make_shared<Callbacks>(); callbacks_->{field} = std::move(value);"
            lines.append(
                f"    {name}& set{method}({signature} value) & {{ {lazy} return *this; }}"
            )
            lines.append(
                f"    {name}&& set{method}({signature} value) && {{ {lazy} return std::move(*this); }}"
            )
    if "pNext" in {member.name for member in struct.members}:
        # No compile-time "extends" constraint here: Vulkan validates the pNext
        # chain at runtime (validation layers / driver). This keeps the chain a
        # single linked list built with `a.setNextInChain(std::move(b.setNextInChain(c)))`.
        lines.append(
            f"    template <typename T> {name}& setNextInChain(T&& value) & {{ nextInChain.set(std::forward<T>(value)); return *this; }}"
        )
        lines.append(
            f"    template <typename T> {name}&& setNextInChain(T&& value) && {{ nextInChain.set(std::forward<T>(value)); return std::move(*this); }}"
        )
    lines.extend(line.rstrip("\r\n") for line in injection)
    lines.extend(
        [
            "    void to_cstruct(CStruct* output) const;",
            f"    void from_cstruct(const native_type& input{_struct_from_parameters(struct, ir, config)});",
            "    void from_output_cstruct(const native_type& input);",
            "};",
        ]
    )
    return _doc_comment(struct.doc or struct.availability.doc, config) + _guard(
        "\n".join(lines), struct.protect or struct.availability.protect
    )


def _emit_structs(
    ir: IrRegistry, config: GeneratorConfig, template: Template
) -> str:
    items = [
        struct
        for struct in ir.structs.values()
        if struct.category == "struct" and struct.active and struct.members
    ]
    names = {struct.name for struct in items}
    emitted: set[str] = set()
    ordered: list[Struct] = []
    while len(ordered) != len(items):
        progress = False
        for struct in items:
            if struct.name in emitted:
                continue
            dependencies: set[str] = set()
            for member in struct.members:
                resolved = ir.resolve(member.type)
                dependency = resolved.name if resolved is not None else member.type
                if dependency not in names or dependency == struct.name:
                    continue
                owns_value = member.pointer_depth == 0 or (
                    member.type in names and member.pointer_depth > 0
                )
                if owns_value:
                    dependencies.add(dependency)
            if dependencies <= emitted:
                ordered.append(struct)
                emitted.add(struct.name)
                progress = True
        if not progress:
            ordered.extend(struct for struct in items if struct.name not in emitted)
            break
    structs = "\n\n".join(
        _emit_struct(
            struct,
            ir,
            config,
            template.injections.get(_cpp_type(struct.name, ir, config), []),
        )
        for struct in ordered
    )
    records: list[str] = []
    for handle in ir.handles.values():
        if not handle.active:
            continue
        alternatives = tuple(
            alt for alt in handle.create_infos if alt in ir.structs
        )
        if len(alternatives) < 2:
            continue
        name = f"{_cpp_type(handle.name, ir, config)}CreationRecord"
        lines = [
            f"struct {name} {{",
            "    using Value = std::variant<",
            "        std::monostate",
        ]
        for alternative in alternatives:
            struct = ir.structs[alternative]
            value = f"        , {_cpp_type(alternative, ir, config)}"
            lines.append(_guard(value, struct.protect or struct.availability.protect))
        lines.extend(["    >;", "    Value value{};", f"    {name}() = default;"])
        for alternative in alternatives:
            struct = ir.structs[alternative]
            cpp = _cpp_type(alternative, ir, config)
            lines.append(
                _guard(
                    f"    {name}(const {cpp}& input) : value(input) {{}}",
                    struct.protect or struct.availability.protect,
                )
            )
        lines.append("};")
        records.append("\n".join(lines))
    return structs + ("\n\n" if structs and records else "") + "\n\n".join(records)


def _to_native_scalar(param: Param, expression: str, ir: IrRegistry) -> str:
    category = _type_category(param.type, ir)
    if category == "handle":
        return f"{expression}.raw()"
    if category in {"enum", "bitmask"}:
        return _native_value(param.type, expression, ir)
    return expression


def _emit_struct_impl(
    struct: Struct, ir: IrRegistry, config: GeneratorConfig
) -> str:
    if not struct.members:
        return ""
    name = _cpp_type(struct.name, ir, config)
    field_names = _struct_member_names(struct)
    callback_groups = _callback_groups(struct, ir)
    callback_members = {member.name for _, members in callback_groups for member in members}
    carrier_members = {carrier for carrier, _ in callback_groups}
    trampolines: list[str] = []
    for carrier, members in callback_groups:
        for member in members:
            func_pointer = ir.func_pointers[member.type]
            field = _callback_field_name(member)
            ret = func_pointer.c_return_type or "void"
            params = ", ".join(p.c_declaration for p in func_pointer.params)
            args, userdata_name = _callback_trampoline_arguments(func_pointer, carrier)
            trampoline_name = f"{name}_{field}_trampoline"
            if ret == "void":
                body = (
                    f"    auto* callbacks = static_cast<{name}::Callbacks*>({userdata_name});\n"
                    f"    if (callbacks && callbacks->{field}) callbacks->{field}({args});"
                )
            else:
                body = (
                    f"    auto* callbacks = static_cast<{name}::Callbacks*>({userdata_name});\n"
                    f"    if (callbacks && callbacks->{field}) return callbacks->{field}({args});\n"
                    "    return {};"
                )
            trampolines.append(
                f"inline VKAPI_ATTR {ret} VKAPI_CALL {trampoline_name}({params}) {{\n{body}\n}}"
            )
    lines = [
        *trampolines,
        f"inline void {name}::to_cstruct(CStruct* output) const {{",
        "    if (!output) return;",
        "    output->value = {};",
    ]
    for member in struct.members:
        category = _type_category(member.type, ir)
        target = f"output->value.{member.name}"
        field = field_names.get(member.name, member.name)
        array_sizes = re.findall(r"\[([^\]]+)\]", member.c_suffix)
        if member.name in carrier_members:
            lines.append(f"    {target} = callbacks_ ? callbacks_.get() : nullptr;")
            continue
        if member.name in callback_members:
            cb_field = _callback_field_name(member)
            lines.append(
                f"    {target} = callbacks_ && callbacks_->{cb_field} ? {name}_{cb_field}_trampoline : nullptr;"
            )
            continue
        if member.name == "pNext":
            lines.append(
                f"    {target} = reinterpret_cast<decltype({target})>(const_cast<void*>(nextInChain.native()));"
            )
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
                    f"    for (std::size_t i = 0; i < {field}.size(); ++i) {target}[i] = {_native_value(member.type, f'{field}[i]', ir)};"
                )
            else:
                lines.append(
                    f"    std::memcpy({target}, {field}.data(), sizeof({target}));"
                )
        elif (
            member.const
            and member.type == "char"
            and member.pointer_depth == 1
            and "null-terminated" in _lengths(member)
        ):
            lines.append(f"    {target} = {field}.c_str();")
        elif _lengths(member) and any(
            length != "null-terminated" for length in _lengths(member)
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
                cpp = _cpp_type(member.type, ir, config)
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
                    else _native_value(member.type, f"{field}[i]", ir)
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
            if member.is_optional:
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
        elif member.pointer_depth == 1 and _is_opaque_raw(member.type, ir):
            lines.append(f"    {target} = {field};")
        elif (
            member.pointer_depth == 1
            and (member.is_optional or category in {"enum", "bitmask"})
            and member.type != "void"
        ):
            lines.extend(
                [
                    f"    output->{member.name}_native = {field} ? std::optional<{_native_type_name(member.type, ir)}>({_native_value(member.type, f'*{field}', ir)}) : std::nullopt;",
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
                f"    {target} = {_to_native_scalar(member, field, ir)};"
            )
    lines.extend(
        [
            "}",
            f"inline void {name}::from_cstruct(const native_type& native{_struct_from_parameters(struct, ir, config)}) {{",
        ]
    )
    for member in struct.members:
        category = _type_category(member.type, ir)
        source = f"native.{member.name}"
        field = field_names.get(member.name, member.name)
        array_sizes = re.findall(r"\[([^\]]+)\]", member.c_suffix)
        if member.name == "pNext":
            continue
        if member.name in carrier_members or member.name in callback_members:
            continue
        if array_sizes:
            if category == "struct":
                nested_args = _nested_from_arguments(member.type, ir, config, struct)
                lines.append(
                    f"    for (std::size_t i = 0; i < {field}.size(); ++i) {field}[i].from_cstruct({source}[i]{nested_args});"
                )
            elif category == "handle":
                lines.append(f"    for (std::size_t i = 0; i < {field}.size(); ++i) {{")
                lines.extend(
                    _borrow_handle_lines(
                        f"{field}[i]", f"{source}[i]", member.type, ir, config, "        "
                    )
                )
                lines.append("    }")
            elif category in {"enum", "bitmask"}:
                lines.append(
                    f"    for (std::size_t i = 0; i < {field}.size(); ++i) {field}[i] = static_cast<{_cpp_type(member.type, ir, config)}>({source}[i]);"
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
            and "null-terminated" in _lengths(member)
        ):
            lines.append(f'    {field} = {source} ? {source} : "";')
            continue
        if _lengths(member) and any(
            length != "null-terminated" for length in _lengths(member)
        ):
            count_source = _native_array_size(member, struct)
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
                nested_args = _nested_from_arguments(member.type, ir, config, struct)
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
                        f"{field}[i]", f"{source}[i]", member.type, ir, config, "            "
                    )
                )
                lines.append("        }")
            elif category in {"enum", "bitmask"}:
                cpp = _cpp_type(member.type, ir, config)
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
            nested_args = _nested_from_arguments(member.type, ir, config, struct)
            if member.is_optional:
                lines.append(
                    f"    if ({source}) {{ {field}.emplace(); {field}->from_cstruct(*{source}{nested_args}); }} else {field}.reset();"
                )
            else:
                lines.append(
                    f"    if ({source}) {field}.from_cstruct(*{source}{nested_args});"
                )
            continue
        if member.pointer_depth == 1 and _is_opaque_raw(member.type, ir):
            lines.append(f"    {field} = {source};")
            continue
        if (
            member.pointer_depth == 1
            and (member.is_optional or category in {"enum", "bitmask"})
            and member.type != "void"
        ):
            conversion = (
                f"static_cast<{_cpp_type(member.type, ir, config)}>(*{source})"
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
                _borrow_handle_lines(field, source, member.type, ir, config)
            )
            continue
        if category in {"enum", "bitmask"}:
            lines.append(
                f"    {field} = static_cast<{_cpp_type(member.type, ir, config)}>({source});"
            )
        elif category == "struct":
            lines.append(
                f"    {field}.from_cstruct({source}{_nested_from_arguments(member.type, ir, config, struct)});"
            )
        else:
            lines.append(f"    {field} = {source};")
    if callback_groups:
        # Reading a native struct back wraps each raw callback + userdata into a
        # forwarder callable so a later to_cstruct can pass them through again.
        native_callbacks = " || ".join(
            f"native.{member.name}"
            for _, members in callback_groups
            for member in members
        )
        lines.append("    callbacks_.reset();")
        lines.append(f"    if ({native_callbacks}) {{")
        lines.append("        callbacks_ = std::make_shared<Callbacks>();")
        for carrier, members in callback_groups:
            for member in members:
                func_pointer = ir.func_pointers[member.type]
                field = _callback_field_name(member)
                ret = func_pointer.c_return_type or "void"
                params = ", ".join(
                    f"{_native_type(p)} {p.name}"
                    for p in func_pointer.params
                    if p.name != carrier
                )
                forwarder_args = ", ".join(
                    "native_userdata" if p.name == carrier else p.name
                    for p in func_pointer.params
                )
                body = (
                    f"{{ native_pfn({forwarder_args}); }}"
                    if ret == "void"
                    else f"{{ return native_pfn({forwarder_args}); }}"
                )
                lines.append(
                    f"        if (native.{member.name}) callbacks_->{field} = [native_pfn = native.{member.name}, native_userdata = native.{carrier}]({params}) -> {ret} {body};"
                )
        lines.append("    }")
    lines.extend(["}"])
    output_lines = [
        f"inline void {name}::from_output_cstruct(const native_type& native) {{"
    ]
    if not _struct_from_parent_types(struct, ir) and not _struct_context_lengths(struct, ir):
        output_lines.append("    from_cstruct(native);")
    else:
        for member in struct.members:
            category = _type_category(member.type, ir)
            if member.name == "pNext" or category == "handle":
                continue
            if member.name in carrier_members or member.name in callback_members:
                continue
            source = f"native.{member.name}"
            field = field_names.get(member.name, member.name)
            array_sizes = re.findall(r"\[([^\]]+)\]", member.c_suffix)
            if array_sizes:
                if category in {"enum", "bitmask"}:
                    cpp = _cpp_type(member.type, ir, config)
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
            elif member.pointer_depth or _lengths(member):
                if not _lengths(member) and member.pointer_depth == 0:
                    output_lines.append(f"    {field} = {source};")
                elif (
                    not _lengths(member)
                    and member.pointer_depth > 0
                    and member.type not in ir.structs
                    and member.type not in ir.handles
                    and _type_category(member.type, ir) is None
                ):
                    output_lines.append(f"    {field} = {source};")
            elif category in {"enum", "bitmask"}:
                output_lines.append(
                    f"    {field} = static_cast<{_cpp_type(member.type, ir, config)}>({source});"
                )
            elif category == "struct":
                output_lines.append(f"    {field}.from_output_cstruct({source});")
            else:
                output_lines.append(f"    {field} = {source};")
    if "pNext" in {member.name for member in struct.members}:
        output_lines.append("    nextInChain.refresh();")
    output_lines.append("}")
    lines.extend(output_lines)
    return _guard("\n".join(lines), struct.protect or struct.availability.protect)


def _emit_struct_implementations(ir: IrRegistry, config: GeneratorConfig) -> str:
    return "\n\n".join(
        _emit_struct_impl(struct, ir, config)
        for struct in ir.structs.values()
        if struct.category == "struct" and struct.active
    )


def _emit_result_code(ir: IrRegistry) -> str:
    group = ir.enums.get("Result")
    if group is None:
        return "enum class ResultCode : std::int32_t { Success = VK_SUCCESS };"
    lines = ["enum class ResultCode : std::int32_t {"]
    used: set[str] = set()
    for value in group.values:
        if not value.active:
            continue
        name = enum_name(group.c_name, value.name, ir.tags)
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


# ---------------------------------------------------------------------------
# Command / handle helpers
# ---------------------------------------------------------------------------

def _receiver_param(command: Command, receiver: str | None) -> Param | None:
    if receiver is None:
        return None
    return next(
        (p for p in command.params if p.type == receiver and p.pointer_depth == 0),
        None,
    )


def _bound_handle_arguments(
    command: Command, receiver: str | None, ir: IrRegistry
) -> dict[int, str]:
    if receiver is None:
        return {}
    expressions: dict[str, str] = {receiver: "raw()"}
    current = _as_handle(ir, receiver)
    chain = ""
    visited: set[str] = set()
    while current and current.parents:
        parent_type = current.parent
        if parent_type in visited:
            break
        visited.add(parent_type)
        chain += ".parent()"
        expressions.setdefault(parent_type, f"(*this){chain}.raw()")
        current = _as_handle(ir, parent_type)
    return {
        id(param): expressions[param.type]
        for param in command.params
        if param.pointer_depth == 0 and param.type in expressions
    }


def _bound_handle_wrapper_arguments(
    command: Command, receiver: str, ir: IrRegistry
) -> dict[int, str]:
    expressions: dict[str, str] = {receiver: "*this"}
    current = _as_handle(ir, receiver)
    expression = "this->parent()"
    visited: set[str] = set()
    while current and current.parents:
        parent_type = current.parent
        if parent_type in visited:
            break
        visited.add(parent_type)
        expressions.setdefault(parent_type, expression)
        expression += ".parent()"
        current = _as_handle(ir, parent_type)
    return {
        id(param): expressions[param.type]
        for param in command.params
        if param.pointer_depth == 0 and param.type in expressions
    }


def _externsync_lines(
    command: Command,
    receiver: str | None,
    ir: IrRegistry,
    result_type: str,
    config: GeneratorConfig,
) -> list[str]:
    if not config.externsync:
        return []
    bound = (
        _bound_handle_wrapper_arguments(command, receiver, ir) if receiver else {}
    )
    targets: list[tuple[str, bool, bool]] = []
    dynamic_targets: list[tuple[str, str]] = []
    for param in command.params:
        if not param.externsync:
            continue
        if _type_category(param.type, ir) != "handle":
            expression = param.externsync or ""
            nested_array = re.fullmatch(
                r"(?:maybe:)?(p[A-Z]\w*)\[\]\.([A-Za-z_]\w*)", expression
            )
            nested_member = re.fullmatch(r"(p[A-Z]\w*)->([A-Za-z_]\w*)", expression)
            struct = _as_struct(ir, param.type)
            if nested_array and struct is not None:
                member = next(
                    (value for value in struct.members if value.name == nested_array.group(2)),
                    None,
                )
                if (
                    member is not None
                    and _type_category(member.type, ir) == "handle"
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
                bool(_lengths(param)) and not is_bound,
                param.is_optional and not is_bound,
            )
        )

    parent_exclusive = (
        bool(command.implicit_externsync)
        and receiver is not None
        and any("commandPool" in text for text in command.implicit_externsync)
    )
    receiver_exclusive = (
        bool(command.implicit_externsync)
        and receiver == "Device"
        and any("VkQueue" in text for text in command.implicit_externsync)
    )
    # Shared (non-exclusive) locks for the receiver and every input handle the
    # command reads without externsyncing it. They serialize against the
    # exclusive externsync locks above; StateLocks dedup prefers exclusive.
    shared_targets: list[tuple[str, bool, bool]] = []
    if receiver is not None:
        shared_targets.append(("*this", False, False))
    for param in command.params:
        if _type_category(param.type, ir) != "handle":
            continue
        if param.externsync:
            continue
        if param.direction != "input":
            continue
        if id(param) in bound:
            continue
        shared_targets.append(
            (
                _public_param_name(param),
                bool(_lengths(param)),
                param.is_optional,
            )
        )
    if (
        not targets
        and not dynamic_targets
        and not parent_exclusive
        and not receiver_exclusive
        and not shared_targets
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
            # Optional by-value handles are plain handle refs (the wrapper's
            # null state represents absence), so no dereference is needed.
            lines.extend(
                [
                    f"if ({expression}) {{",
                    f"    auto lock = detail::ExternsyncAccess::collect({expression}, true, externsync_states);",
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
    for index, (expression, is_span, is_optional) in enumerate(shared_targets):
        if is_span:
            lines.extend(
                [
                    f"for (const auto& value : {expression}) {{",
                    "    auto lock = detail::ExternsyncAccess::collect(value, false, externsync_states);",
                    f"    if (!lock) {{ {failure} }}",
                    "}",
                ]
            )
        elif is_optional:
            lines.extend(
                [
                    f"if ({expression}) {{",
                    f"    auto lock = detail::ExternsyncAccess::collect({expression}, false, externsync_states);",
                    f"    if (!lock) {{ {failure} }}",
                    "}",
                ]
            )
        else:
            lines.extend(
                [
                    f"auto externsync_shared_{index} = detail::ExternsyncAccess::collect({expression}, false, externsync_states);",
                    f"if (!externsync_shared_{index}) {{ auto& lock = externsync_shared_{index}; {failure} }}",
                ]
            )
    lines.append("detail::StateLocks externsync_locks(externsync_states);")
    return lines


def _is_device_scope(handle: Handle, ir: IrRegistry) -> bool:
    current: Handle | None = handle
    visited: set[str] = set()
    while current is not None and current.name not in visited:
        if current.c_name == "VkDevice":
            return True
        visited.add(current.name)
        parent = current.parent
        current = _as_handle(ir, parent) if parent else None
    return False


def _dispatch_function(
    command: Command, receiver: str | None, ir: IrRegistry
) -> str:
    """Resolved PFN expression for a command's dispatch table (no call)."""
    if receiver is None:
        return f"::{command.c_name}"
    if command.c_name == "vkGetInstanceProcAddr":
        return "::vkGetInstanceProcAddr"
    dispatch_type = command.params[0].type if command.params else None
    dispatch_handle = _as_handle(ir, dispatch_type) if dispatch_type else None
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
        if command.c_name == "vkGetDeviceProcAddr"
        or command.c_name in instance_loaded_device_commands
        or dispatch_handle is None
        or not _is_device_scope(dispatch_handle, ir)
        else "device"
    )
    if table == "device":
        return (
            f"(this->dispatchState().device ? this->dispatchState().device->{command.c_name} "
            f": reinterpret_cast<PFN_{command.c_name}>(this->dispatchState().instance->vkGetDeviceProcAddr(this->dispatchState().native_device, \"{command.c_name}\")))"
        )
    # The instance table is always populated by borrow/makeOwned, so there is
    # no global fallback (volk only exports loader-level globals; using
    # ::vkFoo here would not link for table-level instance commands).
    return f"this->dispatchState().instance->{command.c_name}"


def _null_failure(result: str, command: Command) -> str:
    """Return statement emitted when a dispatch slot is null."""
    if result == "void":
        return (
            f"{{ detail::report_error(ResultCode::ErrorExtensionNotPresent, \"{command.cpp_name}\"); return; }}"
        )
    if result.startswith("Result<"):
        return "{ return std::unexpected(ResultCode::ErrorExtensionNotPresent); }"
    if result.startswith("ResultValue<"):
        return f"{{ return {result}{{ResultCode::ErrorExtensionNotPresent, {{}}}}; }}"
    return (
        f"{{ detail::report_error(ResultCode::ErrorExtensionNotPresent, \"{command.cpp_name}\"); return {{}}; }}"
    )


def _dispatch_guard(
    command: Command, receiver: str | None, ir: IrRegistry, result: str
) -> list[str]:
    """Resolve + null-check the dispatch slot before the first call."""
    if receiver is None or command.c_name == "vkGetInstanceProcAddr":
        return []
    function = _dispatch_function(command, receiver, ir)
    return [
        f"auto dispatch_fn = {function};",
        f"if (!dispatch_fn) {_null_failure(result, command)}",
    ]


def _output_handle_parent_expression(
    handle_type: str,
    receiver: str | None,
    command_params: tuple[Param, ...],
    bound: Param | None,
    ir: IrRegistry,
) -> str | None:
    handle = _as_handle(ir, handle_type)
    parent_type = handle.parent if handle else None
    if parent_type is None:
        return None
    if receiver == parent_type:
        return "*this"
    if receiver:
        current = _as_handle(ir, receiver)
        expression = "this->parent()"
        while current and current.parents:
            current_parent = current.parent
            if current_parent == parent_type:
                return expression
            expression += ".parent()"
            current = _as_handle(ir, current_parent)
    for param in command_params:
        if (
            param is not bound
            and param.type == parent_type
            and param.pointer_depth == 0
        ):
            return _public_param_name(param)
    for param in command_params:
        if param is bound:
            continue
        struct = _as_struct(ir, param.type)
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
        if param.pointer_depth and param.is_optional:
            return f"{struct_name}->get().{field_name}"
        return f"{struct_name}.{field_name}"
    return None


def _wrapper_expression_for_type(
    type_name: str,
    receiver: str | None,
    command_params: tuple[Param, ...],
    bound: Param | None,
    ir: IrRegistry,
) -> str | None:
    if receiver:
        current_type = receiver
        expression = "*this"
        while True:
            if current_type == type_name:
                return expression
            current = _as_handle(ir, current_type)
            if current is None or not current.parents:
                break
            current_type = current.parent
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
        struct = _as_struct(ir, param.type)
        if struct is None or struct.category != "struct":
            continue
        struct_expression = _public_param_name(param)
        if param.pointer_depth and param.is_optional:
            struct_expression += "->get()"
        for member in struct.members:
            if (
                member.pointer_depth != 0
                or _type_category(member.type, ir) != "handle"
            ):
                continue
            expression = f"{struct_expression}.{_struct_member_names(struct).get(member.name, member.name)}"
            current_type = member.type
            while True:
                if current_type == type_name:
                    return expression
                current = _as_handle(ir, current_type)
                if current is None or not current.parents:
                    break
                current_type = current.parent
                expression += ".parent()"
    return None


def _command_struct_from_arguments(
    type_name: str,
    receiver: str | None,
    command: Command,
    bound: Param | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> str:
    struct = _as_struct(ir, type_name)
    if not struct:
        return ""
    arguments: list[str] = []
    for parent in _struct_from_parent_types(struct, ir):
        expression = _wrapper_expression_for_type(
            parent, receiver, command.params, bound, ir
        )
        if expression is None:
            raise ValueError(
                f"cannot infer {parent} needed to convert {type_name} returned by {command.c_name}"
            )
        arguments.append(expression)
    return "".join(f", {argument}" for argument in arguments)


def _release_target(command: Command, ir: IrRegistry) -> Param | None:
    if not command.c_name.startswith(("vkDestroy", "vkFree", "vkRelease")):
        return None
    handle_names = ir.handle_names
    handles = [param for param in command.params if param.type in handle_names]
    pointer_targets = [param for param in handles if param.pointer_depth]
    if pointer_targets:
        return pointer_targets[-1] if len(pointer_targets) == 1 else None
    if command.c_name.startswith("vkDestroy"):
        return handles[-1] if handles else None
    if command.c_name.startswith("vkFree"):
        return handles[-1] if len(handles) >= 2 else None
    return handles[-1] if len(handles) >= 2 else None


def _is_owned_handle_output(command: Command, param: Param) -> bool:
    return param.name in command.owned_outputs


def _releaser_command(handle_general: str, ir: IrRegistry) -> Command | None:
    handle = _as_handle(ir, handle_general)
    if handle is None or handle.releaser is None:
        return None
    return ir.commands.get(handle.releaser)


def _handle_release_lambda(
    output: Param,
    producer: Command,
    release: Command,
    receiver: str | None,
    bound: Param | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> str | None:
    handle_names = ir.handle_names
    target = _release_target(release, ir)
    if target is None or target.type != output.type:
        return None
    captures: list[str] = []
    arguments: list[str] = []
    setup: list[str] = []
    used_captures: set[str] = set()
    output_handle = _as_handle(ir, output.type)
    immediate_parent = output_handle.parent if output_handle else None

    def parent_expression(type_name: str) -> str | None:
        if immediate_parent is None:
            return None
        current_type = immediate_parent
        expression = "owner"
        while True:
            if current_type == type_name:
                return expression
            current = _as_handle(ir, current_type)
            if current is None or not current.parents:
                return None
            current_type = current.parent
            expression += ".parent()"

    for param in release.params:
        if param is target:
            arguments.append("&value" if param.pointer_depth else "value")
            continue
        if param.type == "AllocationCallbacks" and param.pointer_depth == 1:
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
                f"{capture} = {public} ? std::optional<{_cpp_type(param.type, ir, config)}>({public}->get()) : std::nullopt"
            )
            setup.extend(
                [
                    f"{_cpp_type(param.type, ir, config)}::CStruct allocator_native{{}};",
                    f"if ({capture}) {capture}->to_cstruct(&allocator_native);",
                ]
            )
            arguments.append(f"{capture} ? &allocator_native.value : nullptr")
            continue
        if param.type in handle_names:
            expression = parent_expression(param.type)
            if expression is None and immediate_parent is None:
                expression = _wrapper_expression_for_type(
                    param.type, receiver, producer.params, bound, ir
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
            param.name in length for length in _lengths(target)
        ):
            arguments.append(f"static_cast<{param.c_type}>(1)")
            continue
        if (
            param.pointer_depth == 0
            and param.name.lower().endswith("count")
            and target.pointer_depth
        ):
            arguments.append(f"static_cast<{param.c_type}>(1)")
            continue
        return None
    dispatch_handle = next(
        (param for param in release.params if param.type in handle_names), None
    )
    dispatch_prefix: str | None = None
    if dispatch_handle is not None:
        dispatch = _as_handle(ir, dispatch_handle.type)
        table = (
            "device"
            if dispatch is not None and _is_device_scope(dispatch, ir)
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
            call = f"release_table.{release.c_name}({', '.join(arguments)})"
        else:
            dispatch_prefix = (
                parent_expression(dispatch_handle.type)
                if immediate_parent is not None
                else "release_" + _public_param_name(dispatch_handle)
            )
        if dispatch_handle is not target and dispatch_prefix:
            call = f"({dispatch_prefix}.dispatchState().{table}->{release.c_name})({', '.join(arguments)})"
        elif dispatch_handle is not target:
            call = f"::{release.c_name}({', '.join(arguments)})"
    else:
        call = f"::{release.c_name}({', '.join(arguments)})"
    body = ["try {", *(f"    {line}" for line in setup)]
    if release.c_return_type == "VkResult":
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
        f"const {_cpp_type(immediate_parent, ir, config)}& owner, "
        if immediate_parent is not None
        else ""
    )
    return f"[{capture_list}]({owner_parameter}{output.c_type} value) noexcept {{ {indented} }}"


def _handle_ownership_condition(
    output: Param,
    producer: Command,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> str | None:
    if output.type != "DescriptorSet" or producer.c_name != "vkAllocateDescriptorSets":
        return None
    allocate_info = next(
        (
            param
            for param in producer.params
            if param.type == "DescriptorSetAllocateInfo"
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
    output: Param,
    command: Command,
    index: str | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> str | None:
    handle = _as_handle(ir, output.type)
    if handle is None or not handle.create_info:
        return None
    record_type = handle.create_info
    create_infos = set(handle.create_infos)
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
    if _lengths(source) and index is not None:
        expression += f"[{index}]"
    elif source.is_optional:
        return None
    record_cpp = _cpp_type(record_type, ir, config)
    return f"std::make_shared<const {record_cpp}>({expression})"


def _command_result_name(command: Command) -> str:
    return command.cpp_name[:1].upper() + command.cpp_name[1:] + "Result"


def _command_parts(
    command: Command,
    receiver: str | None,
    ir: IrRegistry,
    config: GeneratorConfig,
    value_output: Param | tuple[Param, ...] | None = None,
) -> tuple[str, str, str]:
    bound = _receiver_param(command, receiver) if receiver else None
    bound_arguments = (
        _bound_handle_arguments(command, receiver, ir) if receiver else {}
    )
    span_lengths: dict[str, Param] = {}
    for param in command.params:
        if (
            param.const
            and _lengths(param)
            and "null-terminated" not in _lengths(param)
            and param.type != "void"
        ):
            for length in _lengths(param):
                if re.fullmatch(r"[A-Za-z_]\w*", length):
                    span_lengths.setdefault(length, param)

    value_outputs = (
        (value_output,) if isinstance(value_output, Param) else value_output
    ) or ()
    value_output_ids = {id(param) for param in value_outputs}
    visible = [
        param
        for param in command.params
        if id(param) not in bound_arguments
        and param.name not in span_lengths
        and id(param) not in value_output_ids
    ]
    params = ", ".join(
        f"{_public_param_type(param, ir, config)} {_public_param_name(param)}"
        for param in visible
    )
    value_type = None
    status_value_result = False
    if value_outputs:
        if len(value_outputs) == 1:
            cpp = _cpp_type(value_outputs[0].type, ir, config)
            value_type = f"std::vector<{cpp}>" if _lengths(value_outputs[0]) else cpp
        else:
            value_type = _command_result_name(command)
        status_value_result = (
            command.c_return_type == "VkResult" and command.status_alternatives
        )
        if command.c_return_type == "VkResult":
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
            if command.c_return_type == "void"
            else (
                "Result<void>"
                if command.c_return_type == "VkResult"
                else _cpp_type(command.return_type, ir, config)
            )
        )

    prelude: list[str] = []
    postlude: list[str] = []
    failure_cleanup: list[str] = []
    arguments: list[str] = []
    prelude.extend(_externsync_lines(command, receiver, ir, result, config))
    value_locals: dict[int, str] = {}
    if value_outputs:
        if len(value_outputs) > 1:
            prelude.append(f"{value_type} value{{}};")
        for output in value_outputs:
            cpp = _cpp_type(output.type, ir, config)
            local = (
                "value"
                if len(value_outputs) == 1
                else f"result_{_public_param_name(output)}"
            )
            value_locals[id(output)] = local
            if _lengths(output):
                size = _output_size_expression(output, command, ir)
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

    def double_pointer_partition(param: Param) -> tuple[str, str] | None:
        if any(length == "1" for length in _lengths(param)[1:]):
            return None
        outer_lengths = {
            length
            for length in _lengths(param)
            if re.fullmatch(r"[A-Za-z_]\w*", length)
        }
        for candidate in command.params:
            if (
                candidate is param
                or not candidate.const
                or candidate.pointer_depth != 1
            ):
                continue
            if not outer_lengths.intersection(_lengths(candidate)):
                continue
            candidate_struct = _as_struct(ir, candidate.type)
            if candidate_struct is None or candidate_struct.category != "struct":
                continue
            count_member = next(
                (
                    member
                    for member in candidate_struct.members
                    if member.pointer_depth == 0 and member.name.endswith("Count")
                ),
                None,
            )
            if count_member is not None:
                return _public_param_name(candidate), count_member.name
        return None

    def safe_span_count(count_name: str) -> str:
        sources: list[tuple[Param, str]] = []
        for candidate in command.params:
            if count_name not in _lengths(candidate) or not candidate.pointer_depth:
                continue
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
            if not candidate.is_optional and not candidate.no_auto_validity
        ]
        conditional = [
            expression
            for candidate, expression in sources
            if candidate.is_optional or candidate.no_auto_validity
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

    for param in command.params:
        is_value_output = id(param) in value_output_ids
        public = (
            value_locals[id(param)] if is_value_output else _public_param_name(param)
        )
        category = _type_category(param.type, ir)
        if id(param) in bound_arguments:
            arguments.append(bound_arguments[id(param)])
            continue
        if param.name in span_lengths:
            arguments.append(
                f"static_cast<{param.c_type}>({safe_span_count(param.name)})"
            )
            continue

        if (
            param.const
            and param.type == "char"
            and param.pointer_depth == 1
            and "null-terminated" in _lengths(param)
        ):
            if param.is_optional:
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
            bool(_lengths(param)) and "null-terminated" not in _lengths(param)
        ) or "[" in param.c_suffix
        if is_span:
            if param.type == "void":
                arguments.append(
                    f"reinterpret_cast<{'const ' if param.const else ''}void*>({public}.empty() ? nullptr : {public}.data())"
                )
            elif category == "struct":
                cpp = _cpp_type(param.type, ir, config)
                if param.pointer_depth > 1:
                    partition = double_pointer_partition(param)
                    prelude.extend(
                        [
                            f"std::vector<{cpp}::CStruct> {public}_cache({public}.size());",
                            f"std::vector<{param.c_type}> {public}_native({public}.size());",
                            f"for (std::size_t i = 0; i < {public}.size(); ++i) {{ {public}[i].to_cstruct(&{public}_cache[i]); {public}_native[i] = {public}_cache[i].value; }}",
                        ]
                    )
                    if partition is None:
                        prelude.extend(
                            [
                                f"std::vector<const {param.c_type}*> {public}_pointers({public}_native.size());",
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
                                if value_outputs and command.c_return_type == "VkResult"
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
                                f"std::vector<const {param.c_type}*> {public}_pointers({partition_source}_native.size());",
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
                            f"std::vector<{param.c_type}> {public}_native({public}.size());",
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
                            param.type, receiver, command, bound, ir, config
                        )
                        postlude.append(
                            f"for (std::size_t i = 0; i < {public}.size(); ++i) {{ {public}[i].from_cstruct({public}_native[i]{from_args});{_output_chain_refresh(param.type, f'{public}[i]', ir)} }}"
                        )
                    arguments.append(
                        f"{public}_native.empty() ? nullptr : {public}_native.data()"
                    )
            elif category == "handle":
                prelude.append(
                    f"std::vector<{param.c_type}> {public}_native({public}.size());"
                )
                if param.const:
                    prelude.append(
                        f"for (std::size_t i = 0; i < {public}.size(); ++i) {public}_native[i] = {public}[i].raw();"
                    )
                else:
                    cpp = _cpp_type(param.type, ir, config)
                    parent = _output_handle_parent_expression(
                        param.type, receiver, command.params, bound, ir
                    )
                    release = _releaser_command(param.type, ir)
                    owned = release is not None and _is_owned_handle_output(command, param)
                    ownership_condition = (
                        _handle_ownership_condition(param, command, ir, config)
                        if owned
                        else None
                    )
                    destroyer = (
                        _handle_release_lambda(
                            param, command, release, receiver, bound, ir, config
                        )
                        if owned and release is not None
                        else None
                    )
                    record = (
                        _creation_record_expression(param, command, "i", ir, config)
                        if owned
                        else None
                    )
                    if owned and destroyer is None:
                        raise ValueError(
                            f"cannot infer release provenance for {command.c_name}.{param.name}"
                        )
                    if owned:
                        cleanup_call = (
                            f"{public}_cleanup({parent}, native)"
                            if parent
                            else f"{public}_cleanup(native)"
                        )
                        cleanup_lines = [
                            f"auto {public}_cleanup = {destroyer};",
                            f"for (auto native : {public}_native) if (native != {param.c_type}{{}}) {cleanup_call};",
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
                        if ir.handles[param.type].create_info:
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
                            if value_outputs and command.c_return_type == "VkResult"
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
                        cleanup_remainder = f"auto cleanup = {destroyer}; for (std::size_t remaining = i + 1; remaining < {public}_native.size(); ++remaining) if ({public}_native[remaining] != {param.c_type}{{}}) {cleanup_call};"
                        if ownership_condition:
                            cleanup_remainder = (
                                f"if ({ownership_condition}) {{ {cleanup_remainder} }}"
                            )
                        failure = cleanup_remainder + " " + failure
                    postlude.append(
                        f"for (std::size_t i = 0; i < {public}.size(); ++i) {{ if ({public}_native[i] == {param.c_type}{{}}) {{ {public}[i].reset(); continue; }} auto wrapped = {wrap}; if (!wrapped) {{ {failure} }} {public}[i] = std::move(*wrapped); }}"
                    )
                arguments.append(
                    f"{public}_native.empty() ? nullptr : {public}_native.data()"
                )
            elif category in {"enum", "bitmask"}:
                cpp = _cpp_type(param.type, ir, config)
                prelude.append(
                    f"std::vector<{param.c_type}> {public}_native({public}.size());"
                )
                if param.const:
                    prelude.append(
                        f"for (std::size_t i = 0; i < {public}.size(); ++i) {public}_native[i] = {_native_value(param.type, f'{public}[i]', ir)};"
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
                                f"std::vector<const {param.c_type}*> {public}_pointers({public}.size());",
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
                                f"std::vector<const {param.c_type}*> {public}_pointers({partition_source}_native.size());",
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
                arguments.append(f"{public}.raw()")
            elif category in {"enum", "bitmask"}:
                arguments.append(_native_value(param.type, public, ir))
            else:
                arguments.append(public)
            continue

        if param.pointer_depth == 1 and category == "struct":
            cpp = _cpp_type(param.type, ir, config)
            prelude.append(f"{cpp}::CStruct {public}_native{{}};")
            if param.const:
                if param.is_optional:
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
                        param.type, receiver, command, bound, ir, config
                    )
                    postlude.append(
                        f"{public}.from_cstruct({public}_native.value{from_args});{_output_chain_refresh(param.type, public, ir)}"
                    )
                else:
                    prelude.append(
                        f"if ({public}) {public}->to_cstruct(&{public}_native);"
                    )
                    arguments.append(f"{public} ? &{public}_native.value : nullptr")
                    from_args = _command_struct_from_arguments(
                        param.type, receiver, command, bound, ir, config
                    )
                    refresh = (
                        f" {public}->nextInChain.refresh();"
                        if _has_pnext(param.type, ir)
                        else ""
                    )
                    postlude.append(
                        f"if ({public}) {{ {public}->from_cstruct({public}_native.value{from_args});{refresh} }}"
                    )
            continue

        if param.pointer_depth == 1 and category == "handle":
            cpp = _cpp_type(param.type, ir, config)
            if param.const:
                prelude.append(
                    f"{param.c_type} {public}_native = {public} ? {public}->raw() : {param.c_type}{{}};"
                )
                arguments.append(f"&{public}_native")
            else:
                prelude.append(f"{param.c_type} {public}_native{{}};")
                arguments.append(
                    f"&{public}_native"
                    if is_value_output
                    else f"{public} ? &{public}_native : nullptr"
                )
                parent = _output_handle_parent_expression(
                    param.type, receiver, command.params, bound, ir
                )
                release = _releaser_command(param.type, ir)
                owned = release is not None and _is_owned_handle_output(command, param)
                ownership_condition = (
                    _handle_ownership_condition(param, command, ir, config)
                    if owned
                    else None
                )
                destroyer = (
                    _handle_release_lambda(
                        param, command, release, receiver, bound, ir, config
                    )
                    if owned and release is not None
                    else None
                )
                record = (
                    _creation_record_expression(param, command, None, ir, config)
                    if owned
                    else None
                )
                if owned and destroyer is None:
                    raise ValueError(
                        f"cannot infer release provenance for {command.c_name}.{param.name}"
                    )
                if owned:
                    cleanup_call = (
                        f"{public}_cleanup({parent}, {public}_native)"
                        if parent
                        else f"{public}_cleanup({public}_native)"
                    )
                    cleanup_lines = [
                        f"if ({public}_native != {param.c_type}{{}}) {{",
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
                    if ir.handles[param.type].create_info:
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
                        if value_outputs and command.c_return_type == "VkResult"
                        else (
                            "return std::unexpected(wrapped.error());"
                            if result == "Result<void>"
                            else "detail::report_error(wrapped.error());"
                        )
                    )
                )
                if is_value_output:
                    postlude.append(
                        f"if ({public}_native != {param.c_type}{{}}) {{ auto wrapped = {wrap}; if (!wrapped) {{ {failure} }} else {public} = std::move(*wrapped); }}"
                    )
                else:
                    postlude.append(
                        f"if ({public}) {{ if ({public}_native == {param.c_type}{{}}) {public}->reset(); else {{ auto wrapped = {wrap}; if (!wrapped) {{ {failure} }} else *{public} = std::move(*wrapped); }} }}"
                    )
            continue

        if param.pointer_depth == 1 and category in {"enum", "bitmask"}:
            cpp = _cpp_type(param.type, ir, config)
            if param.const:
                prelude.append(
                    f"{param.c_type} {public}_native = {public} ? {_native_value(param.type, f'*{public}', ir)} : {param.c_type}{{}};"
                )
                arguments.append(f"{public} ? &{public}_native : nullptr")
            else:
                prelude.append(f"{param.c_type} {public}_native{{}};")
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
            else _public_argument(param, ir, public)
        )

    if len(value_outputs) > 1:
        postlude.extend(
            f"value.{_public_param_name(output)} = std::move({value_locals[id(output)]});"
            for output in value_outputs
        )

    if command.c_name == "vkCreateDevice":
        # Queues are device-owned (retrieved via vkGetDeviceQueue, not created),
        # so they never got a control block through makeOwned. Create one per
        # queue here, held by the device (raw pointer, released on device
        # finalize) and registered in the device's private-data slot so borrow()
        # finds it. The block's parent is a borrowed Device: it must not retain
        # the device control block, or the device could never reach finalize.
        postlude.extend(
            [
                "if (device && *device) {",
                "    auto queue_dispatch = device->dispatchState().device;",
                "    auto queue_association = device->deviceAssociation();",
                "    if (queue_dispatch && queue_association) {",
                "        for (const auto& queue_info : createInfo.queueCreateInfos) {",
                "            for (uint32_t queue_index = 0; queue_index < queue_info.queueCount; ++queue_index) {",
                "                VkQueue native_queue{};",
                "                queue_dispatch->vkGetDeviceQueue(device_native, queue_info.queueFamilyIndex, queue_index, &native_queue);",
                "                if (native_queue == VkQueue{}) continue;",
                "                auto* queue_state = new (std::nothrow) detail::QueueControlBlock;",
                "                if (!queue_state) continue;",
                "                queue_state->native = native_queue;",
                "                queue_state->parent = Device(device_native, *this);",
                "                queue_state->device_dispatch = queue_dispatch;",
                "                queue_state->native_device = device_native;",
                "                auto register_status = queue_association.dispatch->vkSetPrivateData(queue_association.device, VK_OBJECT_TYPE_QUEUE, detail::raw_key(native_queue), queue_association.slot, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(static_cast<detail::LifetimeHeader*>(queue_state))));",
                "                if (register_status != VK_SUCCESS) { delete queue_state; continue; }",
                "            }",
                "        }",
                "    }",
                "}",
            ]
        )

    if command.c_name == "vkCreateDevice":
        result_values = ir.enums.get("Result")
        overflow_error = (
            "ResultCode::ErrorTooManyObjects"
            if result_values
            and any(value.name == "VK_ERROR_TOO_MANY_OBJECTS" for value in result_values.values)
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
                "auto* vulkan13_features = static_cast<VkPhysicalDeviceVulkan13Features*>(nullptr);",
                "auto* private_data_slots = static_cast<VkDevicePrivateDataCreateInfo*>(nullptr);",
                "for (auto* node = static_cast<const VkBaseInStructure*>(createInfo_native.value.pNext); node; node = node->pNext) {",
                "    if (!private_data_feature && node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES) private_data_feature = const_cast<VkPhysicalDevicePrivateDataFeatures*>(reinterpret_cast<const VkPhysicalDevicePrivateDataFeatures*>(node));",
                "    if (!vulkan13_features && node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) vulkan13_features = const_cast<VkPhysicalDeviceVulkan13Features*>(reinterpret_cast<const VkPhysicalDeviceVulkan13Features*>(node));",
                "    if (!private_data_slots && node->sType == VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO) private_data_slots = const_cast<VkDevicePrivateDataCreateInfo*>(reinterpret_cast<const VkDevicePrivateDataCreateInfo*>(node));",
                "}",
                "VkPhysicalDevicePrivateDataFeatures wrapper_private_data{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES};",
                "if (vulkan13_features) vulkan13_features->privateData = VK_TRUE;",
                "else if (private_data_feature) private_data_feature->privateData = VK_TRUE;",
                "else { wrapper_private_data.privateData = VK_TRUE; wrapper_private_data.pNext = const_cast<void*>(createInfo_native.value.pNext); createInfo_native.value.pNext = &wrapper_private_data; }",
                "VkDevicePrivateDataCreateInfo wrapper_slot_request{VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO};",
                f"if (private_data_slots) {{ if (private_data_slots->privateDataSlotRequestCount == std::numeric_limits<std::uint32_t>::max()) return std::unexpected({overflow_error}); ++private_data_slots->privateDataSlotRequestCount; }}",
                "else { wrapper_slot_request.privateDataSlotRequestCount = 1; wrapper_slot_request.pNext = createInfo_native.value.pNext; createInfo_native.value.pNext = &wrapper_slot_request; }",
            ]
        )

    guard = _dispatch_guard(command, receiver, ir, result)
    call_target = "dispatch_fn" if guard else _dispatch_function(command, receiver, ir)
    call = f"{call_target}({', '.join(arguments)})"
    body = list(prelude)
    body.extend(guard)
    if command.c_return_type == "void":
        body.append(f"{call};")
        body.extend(postlude)
        if value_outputs:
            body.append("return std::move(value);")
    elif command.c_return_type == "VkResult":
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
    command: Command, receiver: str, ir: IrRegistry, config: GeneratorConfig
) -> tuple[str, str, str]:
    return _command_parts(command, receiver, ir, config)


def _method_name(
    command: Command, receiver: str, config: GeneratorConfig
) -> str:
    return command.member_name


def _callable_name(
    command: Command, receiver: str | None, config: GeneratorConfig
) -> str:
    return command.member_name if receiver is not None else command.cpp_name


def _method_decl(
    command: Command, receiver: str, ir: IrRegistry, config: GeneratorConfig
) -> str:
    result, params, _ = _method_parts(command, receiver, ir, config)
    prefix = "" if result == "void" else "[[nodiscard]] "
    return (
        f"    {prefix}{result} {_method_name(command, receiver, config)}({params}) const;"
    )


def _method_impl(
    command: Command, receiver: str, ir: IrRegistry, config: GeneratorConfig
) -> str:
    result, params, body = _method_parts(command, receiver, ir, config)
    if not body:
        return ""
    receiver_name = _cpp_type(receiver, ir, config)
    return f"inline {result} {receiver_name}::{_method_name(command, receiver, config)}({params}) const {{ {body} }}"


def _convenience_parts(
    command: Command,
    receiver: str | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> tuple[str, str, str] | None:
    if (
        command.vector_output is None
        or command.count_param is None
        or command.c_return_type not in {"VkResult", "void"}
    ):
        return None
    is_void = command.c_return_type == "void"
    bound = _receiver_param(command, receiver)
    bound_arguments = _bound_handle_arguments(command, receiver, ir)
    count = command.param(command.count_param)
    vector = command.param(command.vector_output)
    if count is None or vector is None:
        return None
    omitted = {*bound_arguments, id(count), id(vector)}
    retained = [param for param in command.params if id(param) not in omitted]
    params = [
        f"{_public_param_type(param, ir, config)} {_public_param_name(param)}"
        for param in retained
    ]
    count_name = command.count_name or "count"
    params.append(f"std::uint32_t {count_name} = 0")
    value_type = (
        "std::byte"
        if vector.type == "void"
        else _cpp_type(vector.type, ir, config)
    )
    native_value_type = "std::byte" if vector.type == "void" else vector.c_type
    vector_category = _type_category(vector.type, ir)
    result_type = (
        f"std::vector<{value_type}>"
        if is_void
        else (
            f"ResultValue<std::vector<{value_type}>>"
            if command.status_alternatives
            else f"Result<std::vector<{value_type}>>"
        )
    )
    count_type = count.c_type
    prelude: list[str] = []
    postlude: list[str] = []
    arguments: list[str] = []
    for param in command.params:
        public_name = _public_param_name(param)
        if id(param) in bound_arguments:
            arguments.append(bound_arguments[id(param)])
        elif param is count:
            arguments.append("&written" if param.pointer_depth else "written")
        elif param is vector:
            pointer = "native_values.empty() ? nullptr : native_values.data()"
            if param.type == "void":
                pointer = f"reinterpret_cast<void*>({pointer})"
            arguments.append(pointer)
        else:
            category = _type_category(param.type, ir)
            if param.pointer_depth == 0 and category == "handle":
                arguments.append(f"{public_name}.raw()")
            elif param.pointer_depth == 0 and category in {"enum", "bitmask"}:
                arguments.append(_native_value(param.type, public_name, ir))
            elif (
                param.const
                and param.type == "char"
                and param.pointer_depth == 1
                and "null-terminated" in _lengths(param)
            ):
                if param.is_optional:
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
                _lengths(param)
                and "null-terminated" not in _lengths(param)
                and param.type != "void"
            ):
                if category == "struct":
                    cpp = _cpp_type(param.type, ir, config)
                    prelude.extend(
                        [
                            f"std::vector<{cpp}::CStruct> {public_name}_cache({public_name}.size());",
                            f"std::vector<{param.c_type}> {public_name}_native({public_name}.size());",
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
                            param.type, receiver, command, bound, ir, config
                        )
                        postlude.append(
                            f"for (std::size_t i = 0; i < {public_name}.size(); ++i) {{ {public_name}[i].from_cstruct({public_name}_native[i]{from_args});{_output_chain_refresh(param.type, f'{public_name}[i]', ir)} }}"
                        )
                    arguments.append(
                        f"{public_name}_native.empty() ? nullptr : {public_name}_native.data()"
                    )
                elif category == "handle":
                    prelude.extend(
                        [
                            f"std::vector<{param.c_type}> {public_name}_native({public_name}.size());",
                            f"for (std::size_t i = 0; i < {public_name}.size(); ++i) {public_name}_native[i] = {public_name}[i].raw();",
                        ]
                    )
                    arguments.append(
                        f"{public_name}_native.empty() ? nullptr : {public_name}_native.data()"
                    )
                elif category in {"enum", "bitmask"}:
                    prelude.extend(
                        [
                            f"std::vector<{param.c_type}> {public_name}_native({public_name}.size());",
                            f"for (std::size_t i = 0; i < {public_name}.size(); ++i) {public_name}_native[i] = {_native_value(param.type, f'{public_name}[i]', ir)};",
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
                cpp = _cpp_type(param.type, ir, config)
                prelude.append(f"{cpp}::CStruct {public_name}_native{{}};")
                if param.const and param.is_optional:
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
                        param.type, receiver, command, bound, ir, config
                    )
                    refresh = (
                        f" {public_name}->nextInChain.refresh();"
                        if _has_pnext(param.type, ir)
                        else ""
                    )
                    postlude.append(
                        f"if ({public_name}) {{ {public_name}->from_cstruct({public_name}_native.value{from_args});{refresh} }}"
                    )
            else:
                arguments.append(_public_argument(param, ir))
    guard = _dispatch_guard(command, receiver, ir, result_type)
    call_target = "dispatch_fn" if guard else _dispatch_function(command, receiver, ir)
    prelude.extend(guard)
    call = f"{call_target}({', '.join(arguments)})"
    null_arguments = list(arguments)
    null_arguments[command.params.index(vector)] = "nullptr"
    null_call = f"{call_target}({', '.join(null_arguments)})"
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
    if count.pointer_depth:
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
                    if command.status_alternatives
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
            retry_args[command.params.index(count)] = "&required"
            body.extend(
                [
                    f"            status = static_cast<ResultCode>({call_target}({', '.join(retry_args)}));",
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
    if vector.type == "void":
        body.append("        auto values = std::move(native_values);")
    elif category == "struct":
        from_args = _command_struct_from_arguments(
            vector.type, receiver, command, bound, ir, config
        )
        body.extend(
            [
                "        values.resize(native_values.size());",
                f"        for (std::size_t i = 0; i < values.size(); ++i) {{ values[i].from_cstruct(native_values[i]{from_args});{_output_chain_refresh(vector.type, 'values[i]', ir)} }}",
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
        parent_expr = _output_handle_parent_expression(
            vector.type, receiver, command.params, bound, ir
        )
        borrow_args = "native_values[i]" + (f", {parent_expr}" if parent_expr else "")
        body.extend(
            [
                f"        std::vector<{value_type}> values;",
                "        values.reserve(native_values.size());",
                f"        for (std::size_t i = 0; i < native_values.size(); ++i) {{ auto wrapped = {value_type}::borrow({borrow_args}); if (!wrapped) "
                + (
                    f"return ResultValue<std::vector<{value_type}>>{{wrapped.error(), {{}}}};"
                    if command.status_alternatives
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
    elif command.status_alternatives:
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
    command: Command,
    receiver: str | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> str | None:
    parts = _convenience_parts(command, receiver, ir, config)
    if parts is None:
        return None
    result, params, _ = parts
    return f"    [[nodiscard]] {result} {_callable_name(command, receiver, config)}({params}) const;"


def _convenience_impl(
    command: Command,
    receiver: str | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> str | None:
    parts = _convenience_parts(command, receiver, ir, config)
    if parts is None:
        return None
    result, params, body = parts
    params = re.sub(r"\s*=\s*0(?=,|$)", "", params)
    receiver_name = (
        _cpp_type(receiver, ir, config) if receiver is not None else "Context"
    )
    indented = "\n".join(f"    {line}" if line else "" for line in body.splitlines())
    return f"inline {result} {receiver_name}::{_callable_name(command, receiver, config)}({params}) const {{\n{indented}\n}}"


def _output_size_expression(
    output: Param, command: Command, ir: IrRegistry
) -> str | None:
    if not _lengths(output):
        return None
    length = next(
        (value for value in _lengths(output) if value != "null-terminated"), None
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
                and length in _lengths(param)
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
        struct = _as_struct(ir, param.type)
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
    command: Command,
    receiver: str | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> tuple[str, str, str] | None:
    if command.c_return_type not in {"VkResult", "void"}:
        return None
    if command.vector_output is not None:
        return None
    if len(command.outputs) != 1:
        return None
    output = command.param(command.outputs[0])
    if output is None or output.type == "void":
        return None
    if command.c_return_type == "VkResult" and len(command.success_codes) > 1:
        result, params, body = _command_parts(command, receiver, ir, config, output)
        return (result, params, body) if body else None
    bound_arguments = _bound_handle_arguments(command, receiver, ir)
    span_lengths: set[str] = set()
    for param in command.params:
        if param.const and _lengths(param) and "null-terminated" not in _lengths(param):
            span_lengths.update(
                length
                for length in _lengths(param)
                if re.fullmatch(r"[A-Za-z_]\w*", length)
            )
    visible = [
        param
        for param in command.params
        if id(param) not in bound_arguments and param.name not in span_lengths
    ]
    retained = [param for param in visible if param is not output]
    params = ", ".join(
        f"{_public_param_type(param, ir, config)} {_public_param_name(param)}"
        for param in retained
    )
    cpp = _cpp_type(output.type, ir, config)
    size = _output_size_expression(output, command, ir)
    method = _callable_name(command, receiver, config)
    output_argument = "values" if _lengths(output) else "&value"
    call_arguments = ", ".join(
        output_argument if param is output else _public_param_name(param)
        for param in visible
    )
    if _lengths(output):
        if size is None:
            return None
        value_type = f"std::vector<{cpp}>"
        if command.c_return_type == "VkResult":
            result = f"Result<{value_type}>"
            body = (
                f"{value_type} values(static_cast<std::size_t>({size})); auto status = {method}({call_arguments}); "
                "if (!status) return std::unexpected(status.error()); return values;"
            )
        else:
            result = value_type
            body = f"{value_type} values(static_cast<std::size_t>({size})); {method}({call_arguments}); return values;"
    else:
        if command.c_return_type == "VkResult":
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
    command: Command,
    receiver: str | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> str | None:
    parts = _owned_handle_convenience_parts(command, receiver, ir, config)
    if parts is None:
        return None
    result, params, _ = parts
    return f"    [[nodiscard]] {result} {_callable_name(command, receiver, config)}({params}) const;"


def _owned_handle_convenience_impl(
    command: Command,
    receiver: str | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> str | None:
    parts = _owned_handle_convenience_parts(command, receiver, ir, config)
    if parts is None:
        return None
    result, params, body = parts
    receiver_name = (
        _cpp_type(receiver, ir, config) if receiver is not None else "Context"
    )
    return f"inline {result} {receiver_name}::{_callable_name(command, receiver, config)}({params}) const {{ {body} }}"


def _multi_output_parts(
    command: Command,
    receiver: str | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> tuple[str, str, str] | None:
    outputs = tuple(command.param(name) for name in command.outputs)
    outputs = tuple(param for param in outputs if param is not None)
    if (
        command.c_return_type != "VkResult"
        or command.vector_output is not None
        or len(outputs) < 2
    ):
        return None
    if any(
        output.type == "void"
        or (
            _lengths(output)
            and _output_size_expression(output, command, ir) is None
        )
        for output in outputs
    ):
        return None
    result, params, body = _command_parts(command, receiver, ir, config, outputs)
    return (result, params, body) if body else None


def _has_multi_output_result(command: Command, ir: IrRegistry) -> bool:
    outputs = tuple(command.param(name) for name in command.outputs)
    outputs = tuple(param for param in outputs if param is not None)
    return (
        command.c_return_type == "VkResult"
        and command.vector_output is None
        and len(outputs) >= 2
        and all(
            output.type != "void"
            and (
                not _lengths(output)
                or _output_size_expression(output, command, ir) is not None
            )
            for output in outputs
        )
    )


def _multi_output_decl(
    command: Command,
    receiver: str | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> str | None:
    parts = _multi_output_parts(command, receiver, ir, config)
    if parts is None:
        return None
    result, params, _ = parts
    return f"    [[nodiscard]] {result} {_callable_name(command, receiver, config)}({params}) const;"


def _multi_output_impl(
    command: Command,
    receiver: str | None,
    ir: IrRegistry,
    config: GeneratorConfig,
) -> str | None:
    parts = _multi_output_parts(command, receiver, ir, config)
    if parts is None:
        return None
    result, params, body = parts
    receiver_name = (
        _cpp_type(receiver, ir, config) if receiver is not None else "Context"
    )
    return f"inline {result} {receiver_name}::{_callable_name(command, receiver, config)}({params}) const {{ {body} }}"


# ---------------------------------------------------------------------------
# Handle emission
# ---------------------------------------------------------------------------

def _handle_commands(handle: Handle, ir: IrRegistry) -> list[Command]:
    return [
        command
        for command in ir.commands.values()
        if command.active and handle.name in command.receivers
    ]


def _emit_handle(
    handle: Handle,
    ir: IrRegistry,
    config: GeneratorConfig,
    injection: list[str],
    vma_resources: frozenset[str],
) -> str:
    name = _cpp_type(handle.name, ir, config)
    parent = _cpp_type(handle.parent, ir, config) if handle.parent else None
    state_name = f"{name}ControlBlock"
    state_lines = [
        f"namespace detail {{ struct {state_name} final : LifetimeHeader {{",
        f"    using native_type = {handle.c_name};",
        "    inline static std::shared_mutex tracking_mutex;",
        "    native_type native{};",
    ]
    uses_host_registry = (
        not _is_device_scope(handle, ir) or handle.c_name == "VkDevice"
    )
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
    if handle.c_name == "VkInstance":
        state_lines.append("    VolkInstanceTable instance_dispatch{};")
    elif handle.c_name == "VkDevice":
        state_lines.extend(
            [
                "    std::shared_mutex private_data_mutex;",
                "    DeviceAssociation device_association{};",
                "    VolkDeviceTable device_dispatch{};",
            ]
        )
    elif handle.c_name == "VkQueue":
        state_lines.extend(
            [
                "    const VolkDeviceTable* device_dispatch{};",
                "    VkDevice native_device{};",
            ]
        )
    if handle.create_info:
        state_lines.append(
            f"    std::shared_ptr<const {_cpp_type(handle.create_info, ir, config)}> create_info;"
        )
    vma_resource = handle.c_name in vma_resources
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
    lines = ["\n".join(state_lines)]
    handle_doc = _doc_comment(handle.doc or handle.availability.doc, config)
    lines.append(f"{handle_doc}class {name} {{" if handle_doc else f"class {name} {{")
    lines.extend(
        [
            "  public:",
            f"    using native_type = {handle.c_name};",
            "  private:",
            "    native_type native_{};",
        ]
    )
    if parent:
        lines.append(f"    mutable {parent} parent_{{}};")
    else:
        lines.append("    VolkInstanceTable dispatch_{};")
    lines.append(f"    mutable detail::{state_name}* ctrl_{{}};")
    lines.append("    friend struct detail::ExternsyncAccess;")
    lines.append("    friend struct detail::HandleAccess;")
    lines.append(
        "    template <typename Handle> friend bool detail::same_object(const Handle&, const Handle&) noexcept;"
    )
    # Handles navigate each other's private parent/dispatch/association
    # plumbing, so every handle type is a friend of every other handle type.
    for other in ir.handles.values():
        if other.active and other.name != handle.name:
            lines.append(f"    friend class {_cpp_type(other.name, ir, config)};")
    produced_from_context = any(
        command.active
        and not command.receivers
        and any(
            command.param(name) is not None
            and command.param(name).type == handle.name
            and _is_owned_handle_output(command, command.param(name))
            for name in command.outputs
        )
        for command in ir.commands.values()
    )
    if produced_from_context:
        lines.append("    friend class Context;")
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
    if handle.create_info:
        factory_record_arg = f", std::shared_ptr<const {_cpp_type(handle.create_info, ir, config)}> creationRecord"
    lines.append(
        f"    [[nodiscard]] static Result<{name}> makeOwned(native_type native{factory_parent_arg}, "
        f"{factory_destroyer} destroyer{factory_record_arg});"
    )
    # Internal plumbing stays private; other generated handle wrappers reach
    # it through the mutual friend declarations above.
    if handle.c_name == "VkDevice":
        association_expr = (
            "ctrl_ ? ctrl_->device_association : detail::DeviceAssociation{}"
        )
        dispatch_expr = "detail::DispatchState{parent().dispatchState().instance, ctrl_ ? &ctrl_->device_dispatch : nullptr, native_}"
    elif handle.c_name == "VkInstance":
        association_expr = "detail::DeviceAssociation{}"
        dispatch_expr = "detail::DispatchState{ctrl_ ? &ctrl_->instance_dispatch : &dispatch_, nullptr, {}}"
    elif handle.c_name == "VkQueue":
        association_expr = "parent().deviceAssociation()"
        dispatch_expr = (
            "ctrl_ && ctrl_->device_dispatch"
            " ? detail::DispatchState{parent().dispatchState().instance, ctrl_->device_dispatch, ctrl_->native_device}"
            " : parent().dispatchState()"
        )
    else:
        association_expr = (
            "parent().deviceAssociation()"
            if parent and _is_device_scope(handle, ir)
            else "detail::DeviceAssociation{}"
        )
        dispatch_expr = "parent().dispatchState()" if parent else "dispatch_"
    lines.append(
        f"    [[nodiscard]] detail::DeviceAssociation deviceAssociation() const noexcept {{ return {association_expr}; }}"
    )
    if parent:
        lines.append(
            f"    [[nodiscard]] const {parent}& parent() const noexcept {{ return ctrl_ ? ctrl_->parent : parent_; }}"
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
    lines.append(
        f"    [[nodiscard]] detail::DispatchState dispatchState() const noexcept {{ return {dispatch_expr}; }}"
    )
    if handle.create_info:
        cpp_info = _cpp_type(handle.create_info, ir, config)
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
    if handle.create_info:
        cpp_info = _cpp_type(handle.create_info, ir, config)
        create_info_arg = f", std::shared_ptr<const {cpp_info}> creationRecord = {{}}"
    lines.append(
        f"    [[nodiscard]] static Result<{name}> adopt(native_type native{adoption}, std::function<void(native_type)> destroyer{create_info_arg});"
    )
    if vma_resource and parent:
        create_record = (
            f", std::shared_ptr<const {_cpp_type(handle.create_info, ir, config)}> creationRecord"
            if handle.create_info
            else ""
        )
        lines.append(
            f"    [[nodiscard]] static Result<{name}> adoptVma(native_type native, const {parent}& parent, std::shared_ptr<void> allocatorLifetime, VmaAllocator allocator, VmaAllocation allocation, const VmaAllocationInfo& allocationInfo, const VmaAllocationCreateInfo& allocationCreateInfo{create_record});"
        )
    seen: set[tuple[str, str]] = set()
    for command in _handle_commands(handle, ir):
        declaration = _method_decl(command, handle.name, ir, config)
        method_name = _method_name(command, handle.name, config)
        key = (method_name, declaration)
        if key not in seen:
            lines.append(
                _doc_comment(command.doc or command.availability.doc, config, "    ")
                + _guard(
                    declaration,
                    command.protect or command.availability.protect,
                )
            )
            seen.add(key)
        convenience = _convenience_decl(command, handle.name, ir, config)
        if convenience:
            key = (method_name + "#convenience", convenience)
            if key not in seen:
                lines.append(
                    _guard(
                        convenience,
                        command.protect or command.availability.protect,
                    )
                )
                seen.add(key)
        owned_convenience = _owned_handle_convenience_decl(
            command, handle.name, ir, config
        )
        if owned_convenience:
            key = (method_name + "#owned", owned_convenience)
            if key not in seen:
                lines.append(
                    _guard(
                        owned_convenience,
                        command.protect or command.availability.protect,
                    )
                )
                seen.add(key)
        multi_output = _multi_output_decl(command, handle.name, ir, config)
        if multi_output:
            key = (method_name + "#multi", multi_output)
            if key not in seen:
                lines.append(
                    _guard(
                        multi_output,
                        command.protect or command.availability.protect,
                    )
                )
                seen.add(key)
    lines.extend(line.rstrip("\r\n") for line in injection)
    lines.append("};")
    return _guard("\n".join(lines), handle.protect or handle.availability.protect)


def _emit_handles(
    ir: IrRegistry,
    config: GeneratorConfig,
    template: Template,
    vma_resources: frozenset[str],
) -> str:
    active = [handle for handle in ir.handles.values() if handle.active]
    pending = list(active)
    ordered: list[Handle] = []
    emitted: set[str] = set()
    while pending:
        progress = False
        for handle in list(pending):
            parent = handle.parent
            if parent not in ir.handles or parent in emitted:
                ordered.append(handle)
                emitted.add(handle.name)
                pending.remove(handle)
                progress = True
        if not progress:
            ordered.extend(pending)
            break
    handles = "\n\n".join(
        _emit_handle(
            handle,
            ir,
            config,
            template.injections.get(_cpp_type(handle.name, ir, config), []),
            vma_resources,
        )
        for handle in ordered
    )
    return handles + "\n\n" + _emit_context(ir, config)


def _emit_handle_lifetime_impl(
    handle: Handle,
    ir: IrRegistry,
    config: GeneratorConfig,
    vma_resources: frozenset[str],
) -> str:
    name = _cpp_type(handle.name, ir, config)
    state_name = f"{name}ControlBlock"
    parent = _cpp_type(handle.parent, ir, config) if handle.parent else None
    object_type = handle.object_type_enum or "VK_OBJECT_TYPE_UNKNOWN"
    device_scope = _is_device_scope(handle, ir)
    vma_resource = handle.c_name in vma_resources
    uses_host_registry = not device_scope or handle.c_name == "VkDevice"
    lines = [
        f"inline void detail::{state_name}::detach(detail::{state_name}* self) noexcept {{"
    ]
    if device_scope:
        association = (
            "self->device_association"
            if handle.c_name == "VkDevice"
            else "detail::HandleAccess::deviceAssociation(self->parent)"
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
    if handle.c_name == "VkDevice":
        lines.extend(
            [
                "        if (self->create_info && self->device_association) {",
                "            for (const auto& queue_info : self->create_info->queueCreateInfos) {",
                "                for (uint32_t queue_index = 0; queue_index < queue_info.queueCount; ++queue_index) {",
                "                    VkQueue native_queue{};",
                "                    self->device_dispatch.vkGetDeviceQueue(self->native, queue_info.queueFamilyIndex, queue_index, &native_queue);",
                "                    if (native_queue == VkQueue{}) continue;",
                "                    std::uint64_t existing{};",
                "                    self->device_dispatch.vkGetPrivateData(self->native, VK_OBJECT_TYPE_QUEUE, detail::raw_key(native_queue), self->device_association.slot, &existing);",
                "                    if (existing) {",
                "                        auto* state = static_cast<detail::QueueControlBlock*>(reinterpret_cast<detail::LifetimeHeader*>(static_cast<std::uintptr_t>(existing)));",
                "                        state->release(detail::QueueControlBlock::tracking_mutex, state, &detail::QueueControlBlock::detach, &detail::QueueControlBlock::finalize);",
                "                    }",
                "                }",
                "            }",
                "        }",
                "        if (self->device_association) {",
                "            self->device_dispatch.vkDestroyPrivateDataSlot(self->device_association.device, self->device_association.slot, nullptr);",
                "            self->device_association = {};",
                "        }",
            ]
        )
    if vma_resource:
        function = "vmaDestroyBuffer" if handle.c_name == "VkBuffer" else "vmaDestroyImage"
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
    if handle.c_name == "VkDevice":
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
        if handle.c_name == "VkInstance":
            borrow_body += " volkLoadInstanceTable(&value.dispatch_, native);"
        borrow_body += " return value;"
    lines.append(
        f"inline Result<{name}> {name}::borrow(native_type native{adoption}) {{ {borrow_body} }}"
    )
    create_info_arg = ""
    if handle.create_info:
        cpp_info = _cpp_type(handle.create_info, ir, config)
        create_info_arg = f", std::shared_ptr<const {cpp_info}> creationRecord"
    destroyer_type = (
        f"std::function<void({'const ' + parent + '&, ' if parent else ''}native_type)>"
    )
    offered_destroy = "destroyer(parent, native)" if parent else "destroyer(native)"
    factory_parent_arg = f", const {parent}& parent" if parent else ""
    make_lines = [
        f"inline Result<{name}> {name}::makeOwned(native_type native{factory_parent_arg}, {destroyer_type} destroyer{create_info_arg}) {{",
        "    if (!destroyer || native == native_type{}) return std::unexpected(ResultCode::ErrorUnknown);",
    ]
    if device_scope and handle.c_name != "VkDevice":
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
        if handle.c_name == "VkInstance":
            make_lines.append(
                "    volkLoadInstanceTable(&state->instance_dispatch, native);"
            )
        if handle.c_name == "VkDevice":
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
    if handle.create_info:
        make_lines.append("    state->create_info = std::move(creationRecord);")
    make_lines.extend([f"    return {name}(state);", "}"])
    lines.append("\n".join(make_lines))

    adopt_lines = [
        f"inline Result<{name}> {name}::adopt(native_type native{adoption}, std::function<void(native_type)> destroyer{create_info_arg}) {{",
        "    if (!destroyer || native == native_type{}) return std::unexpected(ResultCode::ErrorUnknown);",
    ]
    if device_scope and handle.c_name != "VkDevice":
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
    adopt_record = ", std::move(creationRecord)" if handle.create_info else ""
    parent_arg = ", parent" if parent else ""
    adopt_lines.append(
        f"    return makeOwned(native{parent_arg}{adapter}{adopt_record});"
    )
    adopt_lines.append("}")
    lines.append("\n".join(adopt_lines))
    if vma_resource and parent:
        vma_destroy = "vmaDestroyBuffer" if handle.c_name == "VkBuffer" else "vmaDestroyImage"
        create_record_arg = (
            f", std::shared_ptr<const {_cpp_type(handle.create_info, ir, config)}> creationRecord"
            if handle.create_info
            else ""
        )
        create_record_store = (
            " state->create_info = std::move(creationRecord);"
            if handle.create_info
            else ""
        )
        lines.append(
            f"inline Result<{name}> {name}::adoptVma(native_type native, const {parent}& parent, std::shared_ptr<void> allocatorLifetime, VmaAllocator allocator, VmaAllocation allocation, const VmaAllocationInfo& allocationInfo, const VmaAllocationCreateInfo& allocationCreateInfo{create_record_arg}) {{ if (!allocator || !allocation || native == native_type{{}}) return std::unexpected(ResultCode::ErrorUnknown); auto association = parent.deviceAssociation(); if (!association) {{ {vma_destroy}(allocator, native, allocation); return std::unexpected(ResultCode::ErrorUnknown); }} std::unique_lock lock(detail::{state_name}::tracking_mutex); std::unique_lock association_lock(*association.mutex); std::uint64_t existing{{}}; association.dispatch->vkGetPrivateData(association.device, {object_type}, detail::raw_key(native), association.slot, &existing); detail::{state_name}* state = existing ? static_cast<detail::{state_name}*>(reinterpret_cast<detail::LifetimeHeader*>(static_cast<std::uintptr_t>(existing))) : nullptr; if (state) {{ state->retain(); return {name}(state); }} state = new (std::nothrow) detail::{state_name}; if (!state) {{ association_lock.unlock(); lock.unlock(); {vma_destroy}(allocator, native, allocation); return std::unexpected(ResultCode::ErrorOutOfHostMemory); }} state->native = native; state->parent = parent; state->vma_allocator_lifetime = std::move(allocatorLifetime); state->vma_allocator = allocator; state->vma_allocation = allocation; state->vma_allocation_info = allocationInfo; state->vma_allocation_create_info = allocationCreateInfo;{create_record_store} auto status = association.dispatch->vkSetPrivateData(association.device, {object_type}, detail::raw_key(native), association.slot, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(static_cast<detail::LifetimeHeader*>(state)))); if (status != VK_SUCCESS) {{ association_lock.unlock(); delete state; lock.unlock(); {vma_destroy}(allocator, native, allocation); return std::unexpected(static_cast<ResultCode>(status)); }} return {name}(state); }}"
        )
    return _guard("\n".join(lines), handle.protect or handle.availability.protect)


def _emit_handle_implementations(
    ir: IrRegistry,
    config: GeneratorConfig,
    vma_resources: frozenset[str],
) -> str:
    result: list[str] = []
    for handle in ir.handles.values():
        if not handle.active:
            continue
        result.append(
            _emit_handle_lifetime_impl(handle, ir, config, vma_resources)
        )
        seen: set[str] = set()
        for command in _handle_commands(handle, ir):
            implementation = _method_impl(command, handle.name, ir, config)
            if implementation and implementation not in seen:
                result.append(
                    _guard(
                        implementation,
                        command.protect or command.availability.protect,
                    )
                )
                seen.add(implementation)
            convenience = _convenience_impl(command, handle.name, ir, config)
            if convenience and convenience not in seen:
                result.append(
                    _guard(
                        convenience,
                        command.protect or command.availability.protect,
                    )
                )
                seen.add(convenience)
            owned_convenience = _owned_handle_convenience_impl(
                command, handle.name, ir, config
            )
            if owned_convenience and owned_convenience not in seen:
                result.append(
                    _guard(
                        owned_convenience,
                        command.protect or command.availability.protect,
                    )
                )
                seen.add(owned_convenience)
            multi_output = _multi_output_impl(
                command, handle.name, ir, config
            )
            if multi_output and multi_output not in seen:
                result.append(
                    _guard(
                        multi_output,
                        command.protect or command.availability.protect,
                    )
                )
                seen.add(multi_output)
    result.append(_emit_context_implementations(ir, config))
    return "\n\n".join(result)


def _emit_handle_template_implementations(
    ir: IrRegistry, config: GeneratorConfig
) -> str:
    result: list[str] = []
    for handle in ir.handles.values():
        if not handle.active:
            continue
        name = _cpp_type(handle.name, ir, config)
        definitions = [
            f"template <typename T> inline Result<void> {name}::setData(std::shared_ptr<const T> value) const {{ if (!ctrl_) return std::unexpected(ResultCode::ErrorUnknown); std::unique_lock lock(ctrl_->externsync); try {{ if (!ctrl_->data) ctrl_->data = std::make_unique<std::unordered_map<std::type_index, std::shared_ptr<const void>>>(); ctrl_->data->insert_or_assign(typeid(T), std::move(value)); }} catch (...) {{ return std::unexpected(ResultCode::ErrorOutOfHostMemory); }} return {{}}; }}",
            f"template <typename T> inline std::shared_ptr<const T> {name}::getData() const noexcept {{ if (!ctrl_) return nullptr; std::shared_lock lock(ctrl_->externsync); if (!ctrl_->data) return nullptr; auto found = ctrl_->data->find(typeid(T)); return found == ctrl_->data->end() ? nullptr : std::static_pointer_cast<const T>(found->second); }}",
            f"template <typename T> inline void {name}::clearData() const noexcept {{ if (!ctrl_) return; std::unique_lock lock(ctrl_->externsync); if (ctrl_->data) ctrl_->data->erase(typeid(T)); }}",
        ]
        result.append(
            _guard(
                "\n\n".join(definitions),
                handle.protect or handle.availability.protect,
            )
        )
    return "\n\n".join(result)


def _emit_forwards(ir: IrRegistry, config: GeneratorConfig) -> str:
    values = []
    for struct in ir.structs.values():
        if struct.active and struct.category == "struct" and struct.members:
            values.append(f"struct {_cpp_type(struct.name, ir, config)};")
    for handle in ir.handles.values():
        if not handle.active:
            continue
        values.append(f"class {_cpp_type(handle.name, ir, config)};")
        alternatives = tuple(alt for alt in handle.create_infos if alt in ir.structs)
        if len(alternatives) > 1:
            values.append(
                f"struct {_cpp_type(handle.name, ir, config)}CreationRecord;"
            )
    return "\n".join(values)


def _emit_command_result_forwards(ir: IrRegistry) -> str:
    return "\n".join(
        f"struct {_command_result_name(command)};"
        for command in ir.commands.values()
        if command.active and _has_multi_output_result(command, ir)
    )


def _emit_extensions(ir: IrRegistry, config: GeneratorConfig) -> str:
    lines = []
    for struct in ir.structs.values():
        if not struct.active:
            continue
        extension = _cpp_type(struct.name, ir, config)
        for base in struct.struct_extends:
            lines.append(
                f"template <> struct StructureExtends<{_cpp_type(base, ir, config)}, {extension}> : std::true_type {{}};"
            )
    return "\n".join(lines)


def _context_method_parts(
    command: Command, ir: IrRegistry, config: GeneratorConfig
) -> tuple[str, str, str]:
    return _command_parts(command, None, ir, config)


def _emit_context(ir: IrRegistry, config: GeneratorConfig) -> str:
    commands = [
        command
        for command in ir.commands.values()
        if command.active and not command.receivers
    ]
    version = config.minimum_core.replace(".", "_")
    lines = [
        "// Receiver-less API owner; Context is deliberately not a handle.",
        "class Context {",
        "    Context() noexcept = default;",
        "  public:",
        f"    static constexpr std::uint32_t minimumApiVersion = VK_API_VERSION_{version};",
        "    [[nodiscard]] static Result<Context> create();",
    ]
    for command in commands:
        result, params, _ = _context_method_parts(command, ir, config)
        prefix = "" if result == "void" else "[[nodiscard]] "
        lines.append(f"    {prefix}{result} {command.cpp_name}({params}) const;")
        convenience = _convenience_decl(command, None, ir, config)
        if convenience:
            lines.append(convenience)
        value_convenience = _owned_handle_convenience_decl(command, None, ir, config)
        if value_convenience:
            lines.append(value_convenience)
        multi_output = _multi_output_decl(command, None, ir, config)
        if multi_output:
            lines.append(multi_output)
    lines.append("};")
    return "\n".join(lines)


def _emit_context_implementations(
    ir: IrRegistry, config: GeneratorConfig
) -> str:
    lines: list[str] = [
        "inline Result<Context> Context::create() {",
        "    auto status = volkInitialize();",
        "    if (status != VK_SUCCESS) return std::unexpected(static_cast<ResultCode>(status));",
        "    if (volkGetInstanceVersion() < minimumApiVersion) return std::unexpected(static_cast<ResultCode>(VK_ERROR_INCOMPATIBLE_DRIVER));",
        "    return Context{};",
        "}",
    ]
    for command in ir.commands.values():
        if not command.active or command.receivers:
            continue
        result, params, body = _context_method_parts(command, ir, config)
        if body:
            lines.append(
                _guard(
                    f"inline {result} Context::{command.cpp_name}({params}) const {{ {body} }}",
                    command.protect or command.availability.protect,
                )
            )
        convenience = _convenience_impl(command, None, ir, config)
        if convenience:
            lines.append(
                _guard(
                    convenience,
                    command.protect or command.availability.protect,
                )
            )
        value_convenience = _owned_handle_convenience_impl(command, None, ir, config)
        if value_convenience:
            lines.append(
                _guard(
                    value_convenience,
                    command.protect or command.availability.protect,
                )
            )
        multi_output = _multi_output_impl(command, None, ir, config)
        if multi_output:
            lines.append(
                _guard(
                    multi_output,
                    command.protect or command.availability.protect,
                )
            )
    return "\n\n".join(lines)


def _emit_command_metadata(ir: IrRegistry, config: GeneratorConfig) -> str:
    lines = [
        "// Generated multi-output records and command metadata retained for custom lowering."
    ]
    for command in ir.commands.values():
        if not command.active or not _has_multi_output_result(command, ir):
            continue
        lines.append(f"struct {_command_result_name(command)} {{")
        for name in command.outputs:
            output = command.param(name)
            if output is None:
                continue
            cpp = _cpp_type(output.type, ir, config)
            field_type = f"std::vector<{cpp}>" if _lengths(output) else cpp
            lines.append(f"    {field_type} {_public_param_name(output)}{{}};")
        lines.append("};")
    for command in ir.commands.values():
        if not command.active:
            continue
        successes = ",".join(command.success_codes)
        receivers = ",".join(command.receivers) or "Context"
        outputs = ",".join(command.outputs)
        lines.append(
            f"// {command.c_name}: receivers={receivers}; success={successes}; outputs={outputs}; externsync={','.join(p.name for p in command.params if p.externsync)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runtime prelude (unchanged verbatim from the previous emitter)
# ---------------------------------------------------------------------------

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
struct HandleAccess {
    template <typename Handle>
    [[nodiscard]] static DeviceAssociation deviceAssociation(const Handle& value) noexcept { return value.deviceAssociation(); }
    template <typename Handle>
    [[nodiscard]] static DispatchState dispatchState(const Handle& value) noexcept { return value.dispatchState(); }
    template <typename Handle>
    [[nodiscard]] static const auto& parent(const Handle& value) noexcept { return value.parent(); }
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
};

// Map a bit enum to its mask type; specialized alongside each typed bitmask.
template <typename Bit> struct FlagTraits {};
template <typename Bit> using FlagsOf = Flags<Bit, typename FlagTraits<Bit>::MaskType>;

template <typename Bit>
[[nodiscard]] constexpr FlagsOf<Bit> operator|(Bit lhs, Bit rhs) noexcept { return FlagsOf<Bit>(lhs) | rhs; }
template <typename Bit>
[[nodiscard]] constexpr FlagsOf<Bit> operator|(Bit lhs, FlagsOf<Bit> rhs) noexcept { return FlagsOf<Bit>(lhs) | rhs; }
template <typename Bit>
[[nodiscard]] constexpr FlagsOf<Bit> operator&(Bit lhs, Bit rhs) noexcept { return FlagsOf<Bit>(lhs) & rhs; }
template <typename Bit>
[[nodiscard]] constexpr FlagsOf<Bit> operator&(Bit lhs, FlagsOf<Bit> rhs) noexcept { return FlagsOf<Bit>(lhs) & rhs; }
template <typename Bit>
[[nodiscard]] constexpr FlagsOf<Bit> operator^(Bit lhs, Bit rhs) noexcept { return FlagsOf<Bit>(lhs) ^ rhs; }
template <typename Bit>
[[nodiscard]] constexpr FlagsOf<Bit> operator^(Bit lhs, FlagsOf<Bit> rhs) noexcept { return FlagsOf<Bit>(lhs) ^ rhs; }

template <typename T> using Result = std::expected<T, ResultCode>;
template <typename T> struct ResultValue { ResultCode status{}; T value{}; };
template <typename Base, typename Extension> struct StructureExtends : std::false_type {};
class ExtensionChain {
    struct Value {
        virtual ~Value() = default;
        virtual std::unique_ptr<Value> clone() const = 0;
        virtual const void* native() const = 0;
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
        const void* native() const override {
            // to_cstruct links cache.value.pNext to value.nextInChain.native()
            // recursively, so a chain is a single linked list of native nodes.
            value.to_cstruct(&cache);
            return &cache.value;
        }
        void refresh() override { value.from_output_cstruct(cache.value); }
        [[nodiscard]] std::type_index type() const noexcept override { return typeid(T); }
        [[nodiscard]] void* object() noexcept override { return &value; }
        [[nodiscard]] const void* object() const noexcept override { return &value; }
    };
    std::unique_ptr<Value> value_;
  public:
    ExtensionChain() = default;
    ExtensionChain(const ExtensionChain& rhs) : value_(rhs.value_ ? rhs.value_->clone() : nullptr) {}
    ExtensionChain(ExtensionChain&&) noexcept = default;
    ExtensionChain& operator=(const ExtensionChain& rhs) { if (this != &rhs) value_ = rhs.value_ ? rhs.value_->clone() : nullptr; return *this; }
    ExtensionChain& operator=(ExtensionChain&&) noexcept = default;
    template <typename T> void set(T&& value) { value_ = std::make_unique<Model<std::remove_cvref_t<T>>>(std::forward<T>(value)); }
    void refresh() { if (value_) value_->refresh(); }
    template <typename T> [[nodiscard]] T* get() noexcept {
        return value_ && value_->type() == typeid(T) ? static_cast<T*>(value_->object()) : nullptr;
    }
    template <typename T> [[nodiscard]] const T* get() const noexcept {
        return value_ && value_->type() == typeid(T) ? static_cast<const T*>(value_->object()) : nullptr;
    }
    [[nodiscard]] const void* native() const { return value_ ? value_->native() : nullptr; }
};"""


# ---------------------------------------------------------------------------
# VMA (unchanged, operates on VmaModel)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def emit_sections(
    ir: IrRegistry,
    config: GeneratorConfig,
    template: Template,
    vma: VmaModel | None = None,
) -> dict[str, str]:
    known = {_cpp_type(name, ir, config) for name in ir.type_order}
    unknown_injections = set(template.injections) - known
    if unknown_injections:
        from .template import TemplateError

        raise TemplateError(
            f"injections target unknown types: {', '.join(sorted(unknown_injections))}"
        )
    vma_decl, vma_impl = _vma_sections(vma)
    vma_resources = _vma_resource_types(vma)
    struct_impl = _emit_struct_implementations(ir, config)
    handle_impl = _emit_handle_implementations(ir, config, vma_resources)
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
        "includes": "#ifndef NOMINMAX\n#define NOMINMAX\n#endif\n#include <volk.h>\n"
        + (f"#include <{config.vma_include}>\n" if vma else "")
        + "#include <algorithm>\n#include <array>\n#include <atomic>\n#include <cstdint>\n#include <cstring>\n#include <expected>\n#include <functional>\n#include <limits>\n#include <memory>\n#include <mutex>\n#include <new>\n#include <optional>\n#include <ranges>\n#include <shared_mutex>\n#include <span>\n#include <string>\n#include <string_view>\n#include <typeindex>\n#include <type_traits>\n#include <unordered_map>\n#include <utility>\n#include <variant>\n#include <vector>",
        "forward_declarations": _emit_forwards(ir, config)
        + "\n"
        + _emit_command_result_forwards(ir),
        "result_code": _emit_result_code(ir),
        "aliases": _emit_aliases(ir, config),
        "constants": _emit_constants(ir, config),
        "enums": _emit_enums(ir, config),
        "runtime_declarations": PRELUDE + "\n" + RUNTIME,
        "runtime_implementations": "",
        "structure_extensions": _emit_extensions(ir, config),
        "structs": _emit_structs(ir, config, template),
        "handles": _emit_handles(ir, config, template, vma_resources),
        "context": "",
        "command_declarations": _emit_command_metadata(ir, config),
        "command_implementations": "",
        "struct_implementations": struct_impl,
        "struct_template_implementations": "",
        "handle_implementations": handle_impl,
        "handle_template_implementations": _emit_handle_template_implementations(ir, config),
        "command_template_implementations": "",
        "vma_declarations": vma_decl,
        "vma_implementations": vma_impl,
    }
