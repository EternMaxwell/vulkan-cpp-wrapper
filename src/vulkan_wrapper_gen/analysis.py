from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch

from .config import GeneratorConfig
from .model import Command, Member, Registry, TypeDecl
from .naming import strip_vk, strip_vk_command


@dataclass(slots=True)
class OutputShape:
    outputs: tuple[Member, ...] = ()
    count: Member | None = None
    vector: Member | None = None
    status_value: bool = False
    count_name: str | None = None


@dataclass(slots=True)
class CommandAnalysis:
    command: Command
    cpp_name: str
    receivers: tuple[str, ...]
    output: OutputShape


@dataclass(slots=True)
class HandleAnalysis:
    type: TypeDecl
    cpp_name: str
    create_info: str | None
    commands: list[CommandAnalysis] = field(default_factory=list)


@dataclass(slots=True)
class ApiAnalysis:
    handles: dict[str, HandleAnalysis]
    commands: list[CommandAnalysis]


def _excluded(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def creation_infos_for_handle(registry: Registry, handle: TypeDecl) -> tuple[str, ...]:
    candidates: list[str] = []
    for command in registry.commands.values():
        if command.alias or not command.name.startswith(("vkCreate", "vkAllocate")):
            continue
        output = [p for p in command.params if p.type == handle.name and p.pointer_depth and not p.const]
        if not output:
            continue
        infos = [
            p.type
            for p in command.params
            if p.const
            and p.pointer_depth
            and (
                "CreateInfo" in p.type
                or "AllocateInfo" in p.type
            )
            and registry.types.get(p.type) is not None
            and registry.types[p.type].category == "struct"
        ]
        candidates.extend(infos)
    return tuple(dict.fromkeys(candidates))


def creation_info_for_handle(registry: Registry, handle: TypeDecl) -> str | None:
    candidates = creation_infos_for_handle(registry, handle)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return f"{strip_vk(handle.name)}CreationRecord"
    return None


def release_target(command: Command, handle_types: set[str]) -> Member | None:
    """Return the handle destroyed/freed by a Vulkan lifetime command.

    Dispatch and parent handles precede the released object in registry
    signatures.  Array free commands expose the released handles through the
    sole handle pointer; scalar destroy/free/release commands use the last
    handle parameter.  Commands such as vkReleaseProfilingLockKHR contain no
    releasable handle and are intentionally not classified here.
    """
    if not command.name.startswith(("vkDestroy", "vkFree", "vkRelease")):
        return None
    handles = [param for param in command.params if param.type in handle_types]
    pointer_targets = [param for param in handles if param.pointer_depth]
    if pointer_targets:
        return pointer_targets[-1] if len(pointer_targets) == 1 else None
    if command.name.startswith("vkDestroy"):
        return handles[-1] if handles else None
    if command.name.startswith("vkFree"):
        return handles[-1] if len(handles) >= 2 else None
    # Release commands without an object after their dispatch handle are
    # operational commands, not lifetime endpoints.
    return handles[-1] if len(handles) >= 2 else None


def handle_releasers(registry: Registry, handle_types: set[str]) -> dict[str, Command]:
    result: dict[str, Command] = {}
    for command in registry.commands.values():
        if command.alias:
            continue
        target = release_target(command, handle_types)
        if target is None:
            continue
        previous = result.get(target.type)
        # Prefer Destroy over Free over Release when multiple operational
        # commands mention the same type; this matches Vulkan ownership.
        rank = ("vkDestroy", "vkFree", "vkRelease")
        if previous is None or next(i for i, prefix in enumerate(rank) if command.name.startswith(prefix)) < next(i for i, prefix in enumerate(rank) if previous.name.startswith(prefix)):
            result[target.type] = command
    return result


def is_owned_handle_output(command: Command, output: Member, releasers: dict[str, Command]) -> bool:
    if output.type not in releasers:
        return False
    return command.name.startswith(("vkCreate", "vkAllocate", "vkAcquire", "vkRegister"))


def _output_shape(command: Command, handles: set[str]) -> OutputShape:
    outputs = tuple(
        param for param in command.params
        if param.pointer_depth and not param.const
        and param.name not in {"pUserData"}
        # Vulkan writable outputs conventionally use p/pp prefixes.  A small
        # set of platform APIs take mutable foreign input pointers such as
        # Xlib Display* and Wayland wl_display*; those are not outputs.
        and (param.name.startswith("p") or param.type in handles)
    )
    count: Member | None = None
    vector: Member | None = None
    for output in outputs:
        for length in output.length:
            normalized = length.removeprefix("latexmath:[").removesuffix("]")
            normalized = normalized.removeprefix("p")
            for param in command.params:
                if param.name == length or param.name.lstrip("p") == normalized:
                    count, vector = param, output
                    break
    success = command.success_codes or (("VK_SUCCESS",) if command.return_type == "VkResult" else ())
    count_name = None
    if count is not None:
        count_name = count.name[1:] if count.name.startswith("p") and len(count.name) > 1 and count.name[1].isupper() else count.name
        count_name = count_name[:1].lower() + count_name[1:]
    # A writable count pointer denotes Vulkan's two-call enumeration pattern.
    # A by-value count (for example createInfoCount in
    # vkCreateGraphicsPipelines) instead sizes input/output spans and must not
    # produce an enumeration overload with a user-visible default count.
    if count is not None and count.pointer_depth == 0:
        count = None
        vector = None
        count_name = None
    return OutputShape(outputs, count, vector, len(success) > 1, count_name)


def analyze(registry: Registry, config: GeneratorConfig) -> ApiAnalysis:
    handle_types = {handle.name for handle in registry.handles if not handle.alias and not _excluded(handle.name, config.exclude_types)}
    handles = {
        handle.name: HandleAnalysis(handle, config.type_names.get(handle.name, strip_vk(handle.name)), creation_info_for_handle(registry, handle))
        for handle in registry.handles if handle.name in handle_types
    }
    analyzed: list[CommandAnalysis] = []
    for command in registry.commands.values():
        if _excluded(command.name, config.exclude_commands):
            continue
        lifetime_target = release_target(command, handle_types)
        # Managed handles consume Vulkan Destroy/Free entry points through
        # their exact control-block deleter. Exposing those commands as normal
        # methods would allow destruction behind live wrapper copies.
        if lifetime_target is not None and command.name.startswith(("vkDestroy", "vkFree")):
            continue
        receivers: list[str] = []
        if command.params and command.params[0].type in handle_types:
            receivers.append(command.params[0].type)
        for param in command.params:
            if (
                param.type in handle_types
                and param.pointer_depth == 0
                and "true" not in param.optional
                and param.type not in receivers
            ):
                receivers.append(param.type)
        override = config.receivers.get(command.name)
        if override:
            receivers = [value for value in receivers if value not in override.remove]
            receivers.extend(value for value in override.add if value not in receivers)
        cpp_name = (override.rename if override and override.rename else None) or config.command_names.get(command.name) or strip_vk_command(command.name)
        item = CommandAnalysis(command, cpp_name, tuple(receivers), _output_shape(command, handle_types))
        analyzed.append(item)
        for receiver in receivers:
            if receiver in handles:
                handles[receiver].commands.append(item)
    return ApiAnalysis(handles, analyzed)
