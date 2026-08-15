from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import ConfigError, load_config
from .emitter import emit_sections
from .ir import RegistryError, build_ir
from .naming import strip_vk
from .template import TemplateError, atomic_write, load_template, render_template
from .vma import VmaError, parse_vma_header


def _split_emit_spec(spec: str) -> tuple[Path, Path]:
    """Split an ``--emit`` ``TEMPLATE:OUTPUT`` argument.

    Windows drive letters (``C:``) are tolerated at the start of the template
    path; the separator is the first ``:`` that is not a drive letter.
    """
    for index, char in enumerate(spec):
        if char == ":":
            if index == 1 and spec[0].isalpha():
                continue  # drive letter (e.g. C:\...)
            template, output = spec[:index], spec[index + 1:]
            if not template or not output:
                break
            return Path(template), Path(output)
    raise TemplateError(f"--emit must be TEMPLATE:OUTPUT, got {spec!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vulkan-wrapper-gen",
        description="Generate a configurable C++23 Vulkan wrapper from Khronos registry XML.",
    )
    parser.add_argument("--registry", action="append", type=Path, required=True,
                        help="Registry XML input; repeat for vk.xml, video.xml, and other registries")
    parser.add_argument("--vma-header", type=Path, help="Optional vk_mem_alloc.h parsed with libclang")
    parser.add_argument("--clang-arg", action="append", default=[], help="Argument forwarded to libclang")
    parser.add_argument("--config", type=Path, help="Versioned TOML generator configuration")
    parser.add_argument("--namespace", help="Override the configured C++ namespace")
    parser.add_argument("--emit", action="append", metavar="TEMPLATE:OUTPUT",
                        help="Render TEMPLATE to OUTPUT; repeat for a header/source pair")
    parser.add_argument("--emit-ir", type=Path,
                        help="Write the processed middle-layer IR as JSON and exit")
    parser.add_argument("--check", action="store_true",
                        help="Validate that outputs are current without writing them")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    return parser


def run(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    config = load_config(args.config)
    if args.namespace:
        config.namespace = args.namespace
    if args.emit_ir:
        ir = build_ir(
            args.registry,
            config.api,
            config.include_extensions,
            config.exclude_extensions,
            config,
        )
        known_commands = {command.c_name for command in ir.commands.values()}
        for command, override in config.receivers.items():
            if command not in known_commands and command not in ir.commands:
                raise ConfigError(f"receivers names unknown command {command}")
            for receiver in (*override.add, *override.remove):
                if ir.type_category(strip_vk(receiver)) != "handle":
                    raise ConfigError(f"receivers.{command} names unknown handle type {receiver}")
        for type_name in config.type_names:
            if ir.resolve(strip_vk(type_name)) is None:
                raise ConfigError(f"naming.types names unknown type {type_name}")
        for command_name in config.command_names:
            if command_name not in known_commands and command_name not in ir.commands:
                raise ConfigError(f"naming.commands names unknown command {command_name}")
        atomic_write(args.emit_ir, ir.to_json(indent=2) + "\n")
        print(f"generated {args.emit_ir}")
        return 0
    if not args.emit:
        raise TemplateError("at least one --emit or --emit-ir is required")
    pairs = [_split_emit_spec(spec) for spec in args.emit]
    resolved_outputs = [output.resolve() for _, output in pairs]
    duplicates = sorted({str(path) for path in resolved_outputs if resolved_outputs.count(path) > 1})
    if duplicates:
        raise TemplateError(f"duplicate output paths: {', '.join(duplicates)}")
    ir = build_ir(
        args.registry,
        config.api,
        config.include_extensions,
        config.exclude_extensions,
        config,
    )
    known_commands = {command.c_name for command in ir.commands.values()}
    for command, override in config.receivers.items():
        if command not in known_commands and command not in ir.commands:
            raise ConfigError(f"receivers names unknown command {command}")
        for receiver in (*override.add, *override.remove):
            if ir.type_category(strip_vk(receiver)) != "handle":
                raise ConfigError(f"receivers.{command} names unknown handle type {receiver}")
    for type_name in config.type_names:
        if ir.resolve(strip_vk(type_name)) is None:
            raise ConfigError(f"naming.types names unknown type {type_name}")
    for command_name in config.command_names:
        if command_name not in known_commands and command_name not in ir.commands:
            raise ConfigError(f"naming.commands names unknown command {command_name}")
    vma = parse_vma_header(args.vma_header, tuple(args.clang_arg), config.vma_functions) if args.vma_header else None
    known_types = set(ir.type_order) | set(config.type_names.values())
    rendered_outputs: list[tuple[Path, str]] = []
    # Validate every template before replacing any output. Individual file
    # replacement is atomic; a malformed later template cannot partially
    # update an otherwise paired invocation.
    for template_path, output_path in pairs:
        template = load_template(template_path)
        sections = emit_sections(ir, config, template, vma)
        rendered_outputs.append((output_path, render_template(template, sections, known_types)))
    changed = False
    for output_path, generated in rendered_outputs:
        current = output_path.read_text(encoding="utf-8") if output_path.is_file() else None
        if current != generated:
            changed = True
            if not args.check:
                atomic_write(output_path, generated)
                print(f"generated {output_path}")
    if args.check and changed:
        print("generated outputs are out of date", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except (ConfigError, RegistryError, TemplateError, VmaError) as exc:
        print(f"vulkan-wrapper-gen: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
