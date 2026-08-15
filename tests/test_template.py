from pathlib import Path

import pytest

from vulkan_wrapper_gen.cli import run
from vulkan_wrapper_gen.template import TemplateError, parse_template, render_template


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mini_vk.xml"


def test_injection_blocks_are_removed_and_collected():
    template = parse_template("A\n{{begin_inject}}\ntypename Buffer:\n  void custom();\n{{end_inject}}\n{{body}}\n")
    assert template.text == "A\n{{body}}\n"
    assert template.injections == {"Buffer": ["  void custom();\n"]}
    assert render_template(template, {"body": "B"}, {"Buffer"}) == "A\nB\n"


def test_unknown_marker_and_type_are_errors():
    with pytest.raises(TemplateError, match="unknown template markers"):
        render_template(parse_template("{{missing}}"), {}, set())
    with pytest.raises(TemplateError, match="unknown types"):
        render_template(parse_template("{{begin_inject}}\ntypename Missing:\n x;\n{{end_inject}}"), {}, {"Buffer"})


def test_malformed_injection_is_error():
    with pytest.raises(TemplateError, match="unterminated"):
        parse_template("{{begin_inject}}\ntypename Buffer:\n x;\n")


def test_all_templates_are_validated_before_any_output_is_replaced(tmp_path):
    good = tmp_path / "good.hpp"
    bad = tmp_path / "bad.hpp"
    good.write_text("{{generated_notice}}", encoding="utf-8")
    bad.write_text("{{missing}}", encoding="utf-8")
    first_output = tmp_path / "first.hpp"
    first_output.write_text("unchanged", encoding="utf-8")
    with pytest.raises(TemplateError, match="unknown template markers"):
        run([
            "--registry", str(FIXTURE),
            "--template", str(good), "--output", str(first_output),
            "--template", str(bad), "--output", str(tmp_path / "second.hpp"),
        ])
    assert first_output.read_text(encoding="utf-8") == "unchanged"
