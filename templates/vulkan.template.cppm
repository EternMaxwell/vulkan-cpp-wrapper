module;

{{generated_notice}}
{{includes}}

export module {{module_name}};

{{begin_inject}}
{{end_inject}}

export namespace {{namespace}} {

{{result_code}}

{{runtime_declarations}}
{{forward_declarations}}
{{constants}}
{{enums}}
{{aliases}}
{{structure_extensions}}
{{handles}}
{{structs}}
{{context}}
{{vma_declarations}}
{{command_declarations}}
{{struct_template_implementations}}
{{handle_template_implementations}}
{{command_template_implementations}}

} // namespace {{namespace}}

namespace {{namespace}} {
{{struct_implementations}}
{{handle_implementations}}
{{command_implementations}}
{{vma_implementations}}
} // namespace {{namespace}}
