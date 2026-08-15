# Vulkan C++ Wrapper Generator

`vulkan-wrapper-gen` is a Python 3.12+ generator for configurable C++23 Vulkan wrappers. It reads the Khronos XML registry directly, retains core/extension/alias/platform metadata, and can optionally inspect `vk_mem_alloc.h` with libclang.

The repository contains generator and template sources only. Generated wrappers belong in your build directory and are intentionally ignored by source control.

## Install

```console
python -m pip install -e .
python -m pip install -e ".[vma]"  # only when parsing vk_mem_alloc.h
```

## Generate

Every supplemental registry is explicit. One template maps to one output; repeat both options to render a header/source pair.

```console
vulkan-wrapper-gen \
  --registry /path/to/Vulkan-Headers/registry/vk.xml \
  --registry /path/to/Vulkan-Headers/registry/video.xml \
  --config examples/vulkan-wrapper.toml \
  --template templates/vulkan.template.hpp \
  --output build/generated/vulkan_wrapper.hpp \
  --template templates/vulkan.template.cpp \
  --output build/generated/vulkan_wrapper.cpp
```

Header-only and module presets are available as `templates/vulkan-header-only.template.hpp` and `templates/vulkan.template.cppm`. Add `--vma-header /path/to/vk_mem_alloc.h` and repeat `--clang-arg` for include directories, platform defines, or target flags.

`--check` renders in memory and exits with status 1 if existing outputs differ.

## Templates

Templates use fixed `{{section}}` markers. Unknown or unresolved markers are errors. Custom declarations can be injected into generated types:

```cpp
{{begin_inject}}
typename SwapchainKHR:
    void applicationSpecificMethod() const;
{{end_inject}}
```

The injection target must be a generated C++ type. Declarations and definitions are separate template payloads. `{{structs}}` and `{{handles}}` contain public declarations (with `Context` included in the latter); `{{struct_implementations}}` and `{{handle_implementations}}` contain ordinary definitions. The independently placeable `{{struct_template_implementations}}`, `{{handle_template_implementations}}`, and `{{command_template_implementations}}` sections contain definitions that must remain visible at template instantiation. Struct setters and simple getters intentionally remain inline in their declarations. The complete marker set is visible in the maintained templates.

## Configuration

Configuration is versioned TOML. Filters use shell-style patterns. Per-command receiver overrides can add/remove handle homes or rename methods. The generator binds a command to its dispatch handle and required scalar handle parameters by default; optional synchronization/cache handles are not inferred as receivers.

Generated enums use scoped values and typed `Flags<Bit, NativeMask>` masks. Promoted extension type aliases and safe lower-camel registry constants are retained. Counted queries have span/count overloads plus value-returning overloads; `void` queries return `std::vector<T>`, while `VkResult` queries return `Result<std::vector<T>>` or `ResultValue<std::vector<T>>` when alternative success statuses are meaningful.

Managed wrapper designs target Vulkan 1.3 and private data. Vulkan permits private data only for `VkDevice` and its children, so generated runtime customization uses private-data associations for device objects and host registries for instance-scope objects. Creation-info getters are concrete and non-templated: an eligible `Buffer` exposes `const BufferCreateInfo* createInfo()`, while ambiguous or creation-less handles expose no getter.

Handle wrappers always store their native handle and concrete parent directly. A borrowed wrapper has a null typed control-block pointer and therefore needs no allocation, registry entry, mutex, user-data map, or reference-count operation. Borrowing reuses an existing managed block when private data or the host registry finds one. Adoption and generated creation paths allocate that handle's concrete control-block type; metadata operations may promote a lightweight borrowed value on demand.

Fallible detach/destruction is reported through `setDestructionErrorSink`; destructors and `reset()` remain non-throwing. The sink receives the result, concrete wrapper type name, and native handle identity.

## Middle-layer IR

Registry XML is first transformed into a processed, JSON-serializable
middle-layer IR (`vulkan_wrapper_gen.ir`).  The IR is the generator's single
source of truth: `build_ir` produces it and the C++ emitter consumes it
directly, so receivers, member names, output shapes, creation records and
releasers are read from the IR rather than re-derived from XML.  The IR
normalizes everything the XML leaves implicit while keeping every raw
attribute and doc comment:

- arrays: every `len`/`altlen` reference is resolved to the count parameter it
  sizes (`counts_for`) and back (`lengths`, `is_array`, `is_byte_array`);
  LaTeX lengths keep both their raw text and extracted body;
- naming: every entity collection is keyed by its processed general name
  (`Buffer`, `createBuffer`) and each entity stores `name` (general) plus
  `c_name` (the C API spelling, `VkBuffer`/`vkCreateBuffer`); references to
  other entities (parents, alias targets, bitmask bases, receivers, ...) also
  use general names, while `VK_`/`PFN_`/`StdVideo` names stay verbatim;
- parameters/members: processed names without Vulkan pointer prefixes
  (`public_name`), processed type (`type`) plus exact C type pieces
  (`c_type`/`c_suffix`), pointer depth, constness, optionality, and
  input/output `direction`;
- commands: general name plus C API name, processed return type plus C
  spelling (`return_type`/`c_return_type`), exact reproducible `c_signature`,
  dispatch handle, receivers and per-receiver member names, two-call
  enumeration shape (`count_param`/`vector_output`/`count_name`),
  alternative success statuses, and owned outputs;
- handles: parents, dispatchability, creation records (concrete or
  synthesized variant record), and the matching destroy/free/release command;
- everything else: aliases with resolved categories, enums/bitmasks/basetypes/
  function pointers/defines/constants, foreign platform types, availability,
  platform guards, and comments.

The IR is emitted as JSON with `--emit-ir` and can be consumed by other tools:

```console
vulkan-wrapper-gen \
  --registry /path/to/Vulkan-Headers/registry/vk.xml \
  --registry /path/to/Vulkan-Headers/registry/video.xml \
  --emit-ir build/registry-ir.json
```

## Development

```console
python -m pip install -e ".[test]"
pytest
```
