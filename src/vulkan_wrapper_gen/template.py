from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re
import tempfile


class TemplateError(ValueError):
    pass


@dataclass(slots=True)
class Template:
    text: str
    injections: dict[str, list[str]] = field(default_factory=dict)


_MARKER = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_TYPE = re.compile(r"^\s*typename\s+([A-Za-z_][A-Za-z0-9_]*):\s*$")


def parse_template(text: str) -> Template:
    output: list[str] = []
    injections: dict[str, list[str]] = {}
    inside = False
    current: str | None = None
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        marker = line.strip()
        if marker == "{{begin_inject}}":
            if inside:
                raise TemplateError(f"nested begin_inject at line {line_number}")
            inside, current = True, None
            continue
        if marker == "{{end_inject}}":
            if not inside:
                raise TemplateError(f"end_inject without begin_inject at line {line_number}")
            inside, current = False, None
            continue
        if not inside:
            output.append(line)
            continue
        match = _TYPE.match(line.rstrip("\r\n"))
        if match:
            current = match.group(1)
            if current in injections:
                raise TemplateError(f"duplicate injection for {current} at line {line_number}")
            injections[current] = []
        elif current:
            injections[current].append(line)
        elif line.strip():
            raise TemplateError(f"injection text before a typename at line {line_number}")
    if inside:
        raise TemplateError("unterminated begin_inject block")
    return Template("".join(output), injections)


def render_template(template: Template, sections: dict[str, str], known_types: set[str]) -> str:
    unknown_types = sorted(set(template.injections) - known_types)
    if unknown_types:
        raise TemplateError(f"injections target unknown types: {', '.join(unknown_types)}")
    markers = set(_MARKER.findall(template.text))
    unknown = sorted(markers - set(sections))
    if unknown:
        raise TemplateError(f"unknown template markers: {', '.join(unknown)}")
    rendered = _MARKER.sub(lambda match: sections[match.group(1)], template.text)
    unresolved = sorted(set(_MARKER.findall(rendered)))
    if unresolved:
        raise TemplateError(f"unresolved template markers: {', '.join(unresolved)}")
    return rendered.replace("\r\n", "\n").replace("\r", "\n")


def load_template(path: Path) -> Template:
    try:
        return parse_template(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TemplateError(f"cannot read template {path}: {exc}") from exc


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise

