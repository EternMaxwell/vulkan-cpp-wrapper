from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class VmaError(RuntimeError):
    pass


@dataclass(slots=True)
class VmaParameter:
    name: str
    type: str


@dataclass(slots=True)
class VmaFunction:
    name: str
    return_type: str
    parameters: list[VmaParameter] = field(default_factory=list)


@dataclass(slots=True)
class VmaModel:
    functions: dict[str, VmaFunction] = field(default_factory=dict)
    structs: set[str] = field(default_factory=set)
    handles: set[str] = field(default_factory=set)


def parse_vma_header(path: Path, clang_args: tuple[str, ...] = (), selected: tuple[str, ...] = ()) -> VmaModel:
    if not path.is_file():
        raise VmaError(f"VMA header does not exist: {path}")
    try:
        from clang import cindex
    except ImportError as exc:
        raise VmaError("VMA parsing requires the optional 'libclang' package; install vulkan-wrapper-generator[vma]") from exc
    # VMA tests VMA_IMPLEMENTATION with #ifdef, so defining it to zero still
    # pulls in its implementation and the platform C++ standard library. Leave
    # it undefined when collecting the public declaration surface.
    arguments = ["-x", "c++", "-std=c++17", *clang_args]
    try:
        unit = cindex.Index.create().parse(str(path), args=arguments)
    except Exception as exc:
        raise VmaError(f"libclang could not parse {path}: {exc}") from exc
    errors = [item for item in unit.diagnostics if item.severity >= item.Error]
    if errors:
        formatted = "\n".join(str(item) for item in errors[:20])
        raise VmaError(f"VMA header contains preprocessing/parse errors:\n{formatted}")
    wanted = set(selected)
    model = VmaModel()
    header = path.resolve()
    for cursor in unit.cursor.walk_preorder():
        location = cursor.location.file
        if location is None or Path(location.name).resolve() != header:
            continue
        if cursor.kind == cindex.CursorKind.FUNCTION_DECL and cursor.spelling.startswith("vma"):
            if wanted and cursor.spelling not in wanted:
                continue
            model.functions[cursor.spelling] = VmaFunction(
                cursor.spelling,
                cursor.result_type.spelling,
                [VmaParameter(arg.spelling, arg.type.spelling) for arg in cursor.get_arguments()],
            )
        elif cursor.kind in {cindex.CursorKind.STRUCT_DECL, cindex.CursorKind.TYPEDEF_DECL}:
            if cursor.spelling.startswith("Vma"):
                model.structs.add(cursor.spelling)
                spelling = cursor.type.spelling
                if "*" in spelling:
                    model.handles.add(cursor.spelling)
    missing = sorted(wanted - set(model.functions))
    if missing:
        raise VmaError(f"selected VMA declarations were not found: {', '.join(missing)}")
    return model
