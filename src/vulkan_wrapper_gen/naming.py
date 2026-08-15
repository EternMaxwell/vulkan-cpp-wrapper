from __future__ import annotations

import re


def strip_vk(name: str) -> str:
    return name[2:] if name.startswith("Vk") else name


def strip_vk_command(name: str) -> str:
    name = name[2:] if name.startswith("vk") else name
    return name[:1].lower() + name[1:]


def snake_to_pascal(name: str) -> str:
    return "".join(piece[:1].upper() + piece[1:].lower() for piece in name.split("_") if piece)


def enum_name(group: str, value: str, tags: tuple[str, ...]) -> str:
    raw_group = strip_vk(group)
    # Extension-suffixed groups (VkPresentModeKHR, VkSurfaceTransformFlagBitsKHR)
    # share the prefix/stem of their base group.  Detect and strip the tag so
    # the group prefix and the FlagBits "Bit" suffix are removed correctly.
    base_group = raw_group
    for tag in sorted(tags, key=len, reverse=True):
        if raw_group.endswith(tag):
            base_group = raw_group[: -len(tag)]
            break
    prefixes = [group.upper(), f"VK_{re.sub(r'(?<!^)(?=[A-Z])', '_', base_group).upper()}"]
    flag_bits = re.fullmatch(r"(.+?)FlagBits(\d*)", base_group)
    if flag_bits:
        stem, width_suffix = flag_bits.groups()
        stem = re.sub(r"(?<!^)(?=[A-Z])", "_", stem).upper()
        prefixes.append(f"VK_{stem}{'_' + width_suffix if width_suffix else ''}")
    candidate = value
    for prefix in sorted(set(prefixes), key=len, reverse=True):
        if candidate.startswith(prefix + "_"):
            candidate = candidate[len(prefix) + 1:]
            break
    if candidate.startswith("VK_"):
        candidate = candidate[3:]
    result = snake_to_pascal(candidate)
    for tag in tags:
        if result.endswith(tag.title()) and raw_group.endswith(tag):
            result = result[:-len(tag)]
    if not result:
        result = "None"
    if result[0].isdigit():
        result = "Value" + result
    if flag_bits and result.endswith("Bit"):
        result = result[:-3]
    return result


def flag_mask_name(bits: str) -> str:
    return bits[:-4] + "s" if bits.endswith("Bits") else bits + "Flags"


def constant_name(value: str, tags: tuple[str, ...]) -> str:
    candidate = value.removeprefix("VK_")
    result = snake_to_pascal(candidate)
    for tag in tags:
        titled = tag.title()
        if result.endswith(titled):
            result = result[:-len(titled)] + tag
            break
    result = result[:1].lower() + result[1:]
    if result in {"true", "false"}:
        result += "Value"
    return result
