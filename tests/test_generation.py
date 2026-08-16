from pathlib import Path

from vulkan_wrapper_gen.cli import run
from vulkan_wrapper_gen.config import GeneratorConfig
from vulkan_wrapper_gen.emitter import _cpp_type
from vulkan_wrapper_gen.ir.model import Enum, IrRegistry
from vulkan_wrapper_gen.naming import enum_name

ROOT = Path(__file__).parents[1]


def test_flag_bit_enumerator_names_drop_group_prefix_and_bit_suffix():
    assert (
        enum_name("VkBufferUsageFlagBits", "VK_BUFFER_USAGE_TRANSFER_SRC_BIT", ())
        == "TransferSrc"
    )
    assert (
        enum_name("VkBufferUsageFlagBits2", "VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT", ())
        == "TransferSrc"
    )


def test_enum_group_without_type_declaration_still_uses_cpp_name():
    ir = IrRegistry(enums={"FaultLevel": Enum("FaultLevel", "VkFaultLevel", "enum")})
    assert _cpp_type("FaultLevel", ir, GeneratorConfig()) == "FaultLevel"


def test_supplemental_structs_deep_own_nested_pointers(tmp_path):
    supplemental = tmp_path / "video.xml"
    supplemental.write_text(
        """<registry><types>
      <type category="struct" name="StdVideoLeaf">
        <member><type>uint32_t</type> <name>value</name></member>
      </type>
      <type category="struct" name="StdVideoHrd">
        <member optional="true" len="*_max_sub_layers_minus1 + 1">const <type>StdVideoLeaf</type>* <name>pLayers</name></member>
      </type>
      <type category="struct" name="StdVideoVui">
        <member optional="true">const <type>StdVideoHrd</type>* <name>pHrd</name></member>
      </type>
      <type category="struct" name="StdVideoSps">
        <member><type>uint8_t</type> <name>sps_max_sub_layers_minus1</name></member>
        <member optional="true">const <type>StdVideoVui</type>* <name>pVui</name></member>
      </type>
      <type category="struct" name="VkVideoOwnedInfo">
        <member optional="true">const <type>StdVideoSps</type>* <name>pSps</name></member>
      </type>
    </types></registry>""",
        encoding="utf-8",
    )
    output = tmp_path / "wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--registry",
                str(supplemental),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    hrd = generated[generated.index("struct StdVideoHrd {") :]
    hrd = hrd[: hrd.index("\n};")]
    assert "using native_type = ::StdVideoHrd;" in hrd
    assert "std::vector<StdVideoLeaf> layers{};" in hrd
    assert "::StdVideoLeaf* pLayers" not in hrd
    assert "setPLayers" not in hrd
    assert (
        "void from_cstruct(const native_type& input, std::size_t contextMaxSubLayersMinus1);"
        in hrd
    )
    assert (
        "layers.resize(static_cast<std::size_t>(contextMaxSubLayersMinus1 + 1));"
        in generated
    )
    assert (
        "vui->from_cstruct(*native.pVui, static_cast<std::size_t>(native.sps_max_sub_layers_minus1));"
        in generated
    )
    assert "std::optional<StdVideoSps> sps{};" in generated


def test_array_setters_write_shared_count_field(tmp_path):
    # Vulkan-Hpp keeps every count field explicit; the array setter additionally
    # writes the matching count so the two stay consistent.
    supplemental = tmp_path / "arrays.xml"
    supplemental.write_text(
        """<registry><types>
      <type category="struct" name="VkParallelArrays">
        <member><type>uint32_t</type> <name>valueCount</name></member>
        <member len="valueCount">const <type>uint32_t</type>* <name>pLeft</name></member>
        <member len="valueCount">const <type>uint32_t</type>* <name>pRight</name></member>
      </type>
      <type category="struct" name="VkAlternativeArrays">
        <member><type>uint32_t</type> <name>valueCount</name></member>
        <member optional="true" len="valueCount">const <type>uint32_t</type>* <name>pDirect</name></member>
        <member optional="true,false" len="valueCount">const <type>uint32_t</type>* const* <name>ppIndirect</name></member>
      </type>
    </types></registry>""",
        encoding="utf-8",
    )
    output = tmp_path / "wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--registry",
                str(supplemental),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    parallel = generated[generated.index("struct ParallelArrays {") :]
    parallel = parallel[: parallel.index("\n};")]
    assert "uint32_t valueCount{};" in parallel
    assert "setValueCount(uint32_t value)" in parallel
    assert (
        "setLeft(std::vector<uint32_t> value) & { left = std::move(value); valueCount = static_cast<uint32_t>(left.size()); return *this; }"
        in parallel
    )
    assert (
        "setRight(std::vector<uint32_t> value) & { right = std::move(value); valueCount = static_cast<uint32_t>(right.size()); return *this; }"
        in parallel
    )
    alternative = generated[generated.index("struct AlternativeArrays {") :]
    alternative = alternative[: alternative.index("\n};")]
    assert "uint32_t valueCount{};" in alternative
    assert "setValueCount(uint32_t value)" in alternative
    assert (
        "setDirect(std::vector<uint32_t> value) & { direct = std::move(value); valueCount = static_cast<uint32_t>(direct.size()); return *this; }"
        in alternative
    )
    assert (
        "setIndirect(std::vector<uint32_t> value) & { indirect = std::move(value); valueCount = static_cast<uint32_t>(indirect.size()); return *this; }"
        in alternative
    )
    parallel_impl = generated[generated.index("inline void ParallelArrays::to_cstruct") :]
    parallel_impl = parallel_impl[: parallel_impl.index("\n}")]
    assert "output->value.valueCount = valueCount;" in parallel_impl
    assert "std::size_t capacity" not in parallel_impl


def test_count_members_are_explicit_and_array_setters_write_them(tmp_path):
    # descriptorCount and viewportCount are both plain length fields: they stay
    # explicit real fields, and setImmutableSamplers/setViewports also write them.
    supplemental = tmp_path / "counts.xml"
    supplemental.write_text(
        """<registry><types>
      <type category="struct" name="VkBindingLike">
        <member><type>uint32_t</type> <name>binding</name></member>
        <member optional="true"><type>uint32_t</type> <name>descriptorCount</name></member>
        <member noautovalidity="true" optional="true" len="descriptorCount">const <type>VkBuffer</type>* <name>pImmutableSamplers</name></member>
      </type>
      <type category="struct" name="VkViewportLike">
        <member optional="true"><type>uint32_t</type> <name>viewportCount</name></member>
        <member noautovalidity="true" optional="true" len="viewportCount">const <type>VkBuffer</type>* <name>pViewports</name></member>
      </type>
    </types></registry>""",
        encoding="utf-8",
    )
    output = tmp_path / "wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--registry",
                str(supplemental),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    binding = generated[generated.index("struct BindingLike {") :]
    binding = binding[: binding.index("\n};")]
    assert "uint32_t descriptorCount{};" in binding
    assert "setDescriptorCount(uint32_t value)" in binding
    assert (
        "setImmutableSamplers(std::vector<Buffer> value) & { immutableSamplers = std::move(value); descriptorCount = static_cast<uint32_t>(immutableSamplers.size()); return *this; }"
        in binding
    )
    binding_impl = generated[generated.index("BindingLike::to_cstruct") :]
    binding_impl = binding_impl[: binding_impl.index("\n}")]
    assert "output->value.descriptorCount = descriptorCount;" in binding_impl
    viewport = generated[generated.index("struct ViewportLike {") :]
    viewport = viewport[: viewport.index("\n};")]
    assert "uint32_t viewportCount{};" in viewport
    assert "setViewportCount(uint32_t value)" in viewport
    assert (
        "setViewports(std::vector<Buffer> value) & { viewports = std::move(value); viewportCount = static_cast<uint32_t>(viewports.size()); return *this; }"
        in viewport
    )
    viewport_impl = generated[generated.index("ViewportLike::to_cstruct") :]
    viewport_impl = viewport_impl[: viewport_impl.index("\n}")]
    assert "output->value.viewportCount = viewportCount;" in viewport_impl


def test_callback_members_become_refcounted_callables(tmp_path):
    supplemental = tmp_path / "callbacks.xml"
    supplemental.write_text(
        """<registry><types>
      <type category="funcpointer">
        <proto><type>VkBool32</type> <name>PFN_vkTestCallback</name></proto>
        <param><type>VkBool32</type> <name>flag</name></param>
        <param><type>void</type>* <name>pUserData</name></param>
      </type>
      <type category="struct" name="VkCallbackStruct">
        <member><type>uint32_t</type> <name>valueCount</name></member>
        <member><type>PFN_vkTestCallback</type> <name>pfnCallback</name></member>
        <member optional="true"><type>void</type>* <name>pUserData</name></member>
      </type>
    </types></registry>""",
        encoding="utf-8",
    )
    output = tmp_path / "wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--registry",
                str(supplemental),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    struct_body = generated[generated.index("struct CallbackStruct {") :]
    struct_body = struct_body[: struct_body.index("\n};")]
    assert "struct Callbacks {" in struct_body
    assert "std::function<VkBool32(VkBool32)> callback{};" in struct_body
    assert "std::shared_ptr<Callbacks> callbacks_{};" in struct_body
    assert "setCallback(std::function<VkBool32(VkBool32)> value)" in struct_body
    assert "pfnCallback{}" not in struct_body
    assert "pUserData{}" not in struct_body
    trampoline = generated[generated.index("CallbackStruct_callback_trampoline") :]
    assert "static_cast<CallbackStruct::Callbacks*>(pUserData)" in trampoline
    to_cstruct = generated[generated.index("inline void CallbackStruct::to_cstruct") :]
    to_cstruct = to_cstruct[: to_cstruct.index("\n}")]
    assert (
        "output->value.pfnCallback = callbacks_ && callbacks_->callback ? CallbackStruct_callback_trampoline : nullptr;"
        in to_cstruct
    )
    assert "output->value.pUserData = callbacks_ ? callbacks_.get() : nullptr;" in to_cstruct


def test_multi_callback_struct_shares_one_userdata_carrier(tmp_path):
    # VkAllocationCallbacks-like: several callbacks share one pUserData, with
    # the userdata in different parameter positions and different return types.
    # A funcpointer without a userdata carrier stays a raw field.
    supplemental = tmp_path / "callbacks_multi.xml"
    supplemental.write_text(
        """<registry><types>
      <type category="funcpointer">
        <proto><type>void</type>* <name>PFN_vkAllocLike</name></proto>
        <param><type>void</type>* <name>pUserData</name></param>
        <param><type>size_t</type> <name>size</name></param>
        <param><type>size_t</type> <name>alignment</name></param>
      </type>
      <type category="funcpointer">
        <proto><type>void</type> <name>PFN_vkFreeLike</name></proto>
        <param><type>void</type>* <name>pUserData</name></param>
        <param><type>uint32_t</type> <name>handle</name></param>
      </type>
      <type category="funcpointer">
        <proto><type>VkBool32</type> <name>PFN_vkNotifyLike</name></proto>
        <param><type>uint32_t</type> <name>flags</name></param>
        <param><type>void</type>* <name>pUserData</name></param>
      </type>
      <type category="struct" name="VkMultiCallbackStruct">
        <member><type>uint32_t</type> <name>valueCount</name></member>
        <member optional="true"><type>void</type>* <name>pUserData</name></member>
        <member><type>PFN_vkAllocLike</type> <name>pfnAlloc</name></member>
        <member><type>PFN_vkFreeLike</type> <name>pfnFree</name></member>
        <member><type>PFN_vkNotifyLike</type> <name>pfnNotify</name></member>
      </type>
      <type category="funcpointer">
        <proto><type>void</type> <name>PFN_vkPlainSlot</name></proto>
        <param><type>uint32_t</type> <name>value</name></param>
      </type>
      <type category="struct" name="VkPlainFuncPtrStruct">
        <member><type>PFN_vkPlainSlot</type> <name>pfnSlot</name></member>
      </type>
    </types></registry>""",
        encoding="utf-8",
    )
    output = tmp_path / "wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--registry",
                str(supplemental),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    body = generated[generated.index("struct MultiCallbackStruct {") :]
    body = body[: body.index("\n};")]
    assert "struct Callbacks {" in body
    assert "std::function<void*(size_t, size_t)> alloc{};" in body
    assert "std::function<void(uint32_t)> free{};" in body
    assert "std::function<VkBool32(uint32_t)> notify{};" in body
    assert "std::shared_ptr<Callbacks> callbacks_{};" in body
    assert "setAlloc(std::function<void*(size_t, size_t)> value)" in body
    assert "setFree(std::function<void(uint32_t)> value)" in body
    assert "setNotify(std::function<VkBool32(uint32_t)> value)" in body
    assert "pUserData{}" not in body
    assert "pfnAlloc{}" not in body
    assert "pfnFree{}" not in body
    assert "pfnNotify{}" not in body
    # userdata-first callback (void* return) drops pUserData from position 0.
    alloc_tramp = generated[
        generated.index("MultiCallbackStruct_alloc_trampoline") :
    ]
    assert (
        "if (callbacks && callbacks->alloc) return callbacks->alloc(size, alignment);"
        in alloc_tramp
    )
    # userdata-last callback drops pUserData from the tail.
    notify_tramp = generated[
        generated.index("MultiCallbackStruct_notify_trampoline") :
    ]
    assert (
        "if (callbacks && callbacks->notify) return callbacks->notify(flags);"
        in notify_tramp
    )
    to_cstruct = generated[generated.index("inline void MultiCallbackStruct::to_cstruct") :]
    to_cstruct = to_cstruct[: to_cstruct.index("\n}")]
    assert (
        "output->value.pfnAlloc = callbacks_ && callbacks_->alloc ? MultiCallbackStruct_alloc_trampoline : nullptr;"
        in to_cstruct
    )
    assert (
        "output->value.pfnFree = callbacks_ && callbacks_->free ? MultiCallbackStruct_free_trampoline : nullptr;"
        in to_cstruct
    )
    assert (
        "output->value.pfnNotify = callbacks_ && callbacks_->notify ? MultiCallbackStruct_notify_trampoline : nullptr;"
        in to_cstruct
    )
    assert "output->value.pUserData = callbacks_ ? callbacks_.get() : nullptr;" in to_cstruct
    # A funcpointer with no userdata carrier stays a plain raw field.
    plain = generated[generated.index("struct PlainFuncPtrStruct {") :]
    plain = plain[: plain.index("\n};")]
    assert "struct Callbacks" not in plain
    assert "PFN_vkPlainSlot pfnSlot{};" in plain
    assert "setPfnSlot(PFN_vkPlainSlot value)" in plain


def test_shared_command_counts_never_exceed_span_storage(tmp_path):
    supplemental = tmp_path / "commands.xml"
    supplemental.write_text(
        """<registry><commands>
      <command successcodes="VK_SUCCESS">
        <proto><type>VkResult</type> <name>vkParallelSpansEXT</name></proto>
        <param><type>VkDevice</type> <name>device</name></param>
        <param><type>uint32_t</type> <name>valueCount</name></param>
        <param len="valueCount">const <type>uint32_t</type>* <name>pLeft</name></param>
        <param len="valueCount">const <type>uint32_t</type>* <name>pRight</name></param>
        <param optional="true" len="valueCount">const <type>uint32_t</type>* <name>pOptional</name></param>
      </command>
    </commands></registry>""",
        encoding="utf-8",
    )
    output = tmp_path / "wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--registry",
                str(supplemental),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    implementation = generated[generated.index("Device::parallelSpansEXT") :]
    implementation = implementation[: implementation.index("\n}")]
    assert "std::size_t capacity = left.size();" in implementation
    assert "right.size()" in implementation
    assert "optional.size()" in implementation
    assert "candidate != 0 && candidate < capacity" in implementation
    assert "optional.empty() ? nullptr : optional.data()" in implementation


def test_single_success_multi_output_uses_expected_value(tmp_path):
    supplemental = tmp_path / "multi-output.xml"
    supplemental.write_text(
        """<registry><commands>
      <command successcodes="VK_SUCCESS" errorcodes="VK_ERROR_UNKNOWN">
        <proto><type>VkResult</type> <name>vkGetPairEXT</name></proto>
        <param><type>VkDevice</type> <name>device</name></param>
        <param><type>uint32_t</type>* <name>pLeft</name></param>
        <param><type>uint64_t</type>* <name>pRight</name></param>
      </command>
    </commands></registry>""",
        encoding="utf-8",
    )
    output = tmp_path / "wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--registry",
                str(supplemental),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    assert "Result<GetPairEXTResult> getPairEXT() const;" in generated
    assert "ResultValue<GetPairEXTResult> getPairEXT() const;" not in generated
    implementation = generated[
        generated.index("inline Result<GetPairEXTResult> Device::getPairEXT") :
    ]
    assert "return std::unexpected(status);" in implementation
    assert "return std::move(value);" in implementation


def test_returned_struct_handles_use_concrete_parent_conversion(tmp_path):
    headers = Path.home() / "AppData" / "Local" / "Temp" / "Vulkan-Headers"
    if not headers.is_dir():
        return
    output = tmp_path / "wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(headers / "registry" / "vk.xml"),
                "--registry",
                str(headers / "registry" / "video.xml"),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    assert (
        "void DisplayPropertiesKHR::from_cstruct(const native_type& native, const PhysicalDevice& ownerPhysicalDevice)"
        in generated
    )
    assert "DisplayKHR::borrow(native.display, ownerPhysicalDevice)" in generated
    assert "values[i].from_cstruct(native_values[i], *this);" in generated
    assert "template <typename Owner>" not in generated
    assert (
        "for (std::size_t remaining = i + 1; remaining < pipelines_native.size(); ++remaining)"
        in generated
    )
    assert "++private_data_slots->privateDataSlotRequestCount" in generated
    assert (
        "private_data_slots->privateDataSlotRequestCount == std::numeric_limits<std::uint32_t>::max()"
        in generated
    )
    assert (
        "if (!private_data_slots && node->sType == VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO)"
        in generated
    )
    pipeline = generated[generated.index("class Pipeline {") :]
    pipeline = pipeline[: pipeline.index("\n};")]
    assert "const PipelineCreationRecord* createInfo() const noexcept" in pipeline
    assert "struct PipelineCreationRecord;" in generated
    assert "std::make_shared<const PipelineCreationRecord>(createInfos[i])" in generated
    allocate_sets = generated[
        generated.index("inline Result<void> Device::allocateDescriptorSets") :
    ]
    allocate_sets = allocate_sets[
        : allocate_sets.index(
            "inline Result<std::vector<DescriptorSet>> Device::allocateDescriptorSets"
        )
    ]
    free_condition = (
        "allocateInfo.descriptorPool.createInfo() && "
        "allocateInfo.descriptorPool.createInfo()->flags.test("
        "DescriptorPoolCreateFlagBits::FreeDescriptorSet)"
    )
    assert free_condition in allocate_sets
    assert f"{free_condition} ? DescriptorSet::makeOwned(" in allocate_sets
    assert ": DescriptorSet::borrow(" in allocate_sets
    descriptor_set_class = generated[generated.index("class DescriptorSet {") :]
    descriptor_set_class = descriptor_set_class[: descriptor_set_class.index("\n};")]
    public_descriptor_set = descriptor_set_class.rsplit("  public:", 1)[1]
    assert "makeOwned(" not in public_descriptor_set
    assert (
        "std::function<void(const DescriptorPool&, native_type)>"
        not in public_descriptor_set
    )


def test_registry_constants_receive_safe_cpp_names(tmp_path):
    output = tmp_path / "constants.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    assert (
        "inline constexpr uint32_t maxTest = static_cast<uint32_t>(VK_MAX_TEST);"
        in generated
    )
    assert "using BufferEXT = Buffer;" in generated
    assert "using BufferCreateInfoEXT = BufferCreateInfo;" in generated
    assert (
        "std::vector<SubmitInfo> getSubmitInfos(std::uint32_t submitCount = 0) const;"
        in generated
    )
    assert "BufferCreateInfo&& setNextInChain(T&& value) &&" in generated
    assert "std::unique_ptr<Value> value_;" in generated
    assert "value_ = std::make_unique<Model" in generated
    assert "void refresh() { if (value_) value_->refresh(); }" in generated
    assert "template <typename T> [[nodiscard]] T* get() noexcept" in generated
    assert "template <typename T> [[nodiscard]] const T* get() const noexcept" in generated
    shader = generated[generated.index("struct ShaderModuleCreateInfo {") :]
    shader = shader[: shader.index("\n};")]
    assert "size_t codeSize{};" in shader
    assert "std::vector<uint32_t> code{};" in shader
    assert "setCodeSize(size_t value)" in shader
    assert "setCode(std::vector<uint32_t> value)" in shader
    assert "codeSize = static_cast<size_t>(code.size() * 4);" in shader
    assert (
        "code.assign(native.pCode, native.pCode + static_cast<std::size_t>(native.codeSize / 4));"
        in generated
    )
    assert (
        "enumerateInstanceExtensionProperties(std::optional<std::string_view> layerName"
        in generated
    )
    assert "std::optional<std::string> layerName_native" in generated
    assert "layerName_native ? layerName_native->c_str() : nullptr" in generated


def test_externsync_shared_locks_and_config_option(tmp_path):
    output = tmp_path / "wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    use_buffer = generated[generated.index("inline void Buffer::useEXT") :]
    use_buffer = use_buffer[: use_buffer.index("\n}")]
    # The externsync'd buffer (bound to *this) gets an exclusive lock; the
    # non-externsync input handle (fence) gets a shared lock so it serializes
    # against the exclusive one.
    assert "collect(*this, true, externsync_states)" in use_buffer
    assert "collect(fence, false, externsync_states)" in use_buffer

    # Disabling externsync in the config drops the lock machinery from commands.
    config = tmp_path / "no-externsync.toml"
    config.write_text("version = 1\n[generator]\nexternsync = false\n", encoding="utf-8")
    plain = tmp_path / "plain.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--config",
                str(config),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(plain),
            ]
        )
        == 0
    )
    plain_text = plain.read_text(encoding="utf-8")
    assert "externsync_states" not in plain_text
    assert "StateLocks externsync_locks" not in plain_text

    # The CLI flag is equivalent to the config option.
    flagged = tmp_path / "flagged.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--set", "externsync=false",
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(flagged),
            ]
        )
        == 0
    )
    flagged_text = flagged.read_text(encoding="utf-8")
    assert "externsync_states" not in flagged_text


def test_cli_set_and_add_config_overrides(tmp_path):
    output = tmp_path / "wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--set", "minimum_core=1.2",
                "--add", "exclude_commands=vkCreateBuffer",
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    # --add appends to the exclusion list; vkCreateBuffer disappears.
    assert "createBuffer" not in generated
    assert "vkDestroyBuffer" not in generated


def test_commands_rehome_to_owned_second_handle(tmp_path):
    output = tmp_path / "wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
            ]
        )
        == 0
    )
    generated = output.read_text(encoding="utf-8")
    # vkUseBufferEXT(device, buffer, fence): buffer is a non-optional handle
    # owned by device, so the command rehomes onto Buffer (Vulkan-Hpp rule).
    assert "void Buffer::useEXT(" in generated
    assert "void Device::useBufferEXT(" not in generated
    # vkCreateGraphicsPipelines keeps pipelineCache optional, so it stays on the
    # dispatch handle.
    assert "Device::createGraphicsPipelines" in generated
    assert "PipelineCache::createGraphicsPipelines" not in generated
    # vkQueueSubmit's first parameter is the queue itself, so no rehome.
    assert "Queue::submit(" in generated


FIXTURE = ROOT / "tests" / "fixtures" / "mini_vk.xml"


def test_header_only_generation_is_deterministic(tmp_path):
    output = tmp_path / "wrapper.hpp"
    arguments = [
        "--registry",
        str(FIXTURE),
        "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
    ]
    assert run(arguments) == 0
    first = output.read_text(encoding="utf-8")
    assert "const BufferCreateInfo* createInfo() const noexcept" in first
    assert "struct BufferControlBlock final" in first
    assert "struct BufferControlBlock final : LifetimeHeader" in first
    lifetime = first[
        first.index("struct LifetimeHeader") : first.index("struct DeviceAssociation")
    ]
    assert "void* owner" not in lifetime
    assert "final_release" not in lifetime
    assert "\n    void (*detach)(void*)" not in lifetime
    assert "std::shared_ptr<const BufferCreateInfo> create_info" in first
    assert "std::shared_ptr<const void> create_info" not in first
    assert "next_identity" not in first
    assert "ObjectKey" not in first
    assert "ObjectKeyHash" not in first
    assert "managed_state" not in first
    assert "adopt_state" not in first
    assert "bool owning" not in first
    assert "HostRegistry" not in first
    assert "controlBlock()" not in first
    assert "registry_mutex" not in first
    assert "OwnerClaim" not in first
    assert "claim_owner_control_block" not in first
    assert "ensure_observer_control_block" not in first
    assert "adoptManaged(" not in first
    assert "state->retain(); return Buffer(state); }" in first
    assert "return makeOwned(native, parent," in first
    assert "fromState(" not in first
    assert "reinterpret_cast<std::uintptr_t>(ctrl_)" in first
    assert "native_type native_{}" in first
    assert "mutable detail::BufferControlBlock* ctrl_{}" in first
    instance_state = first[first.index("struct InstanceControlBlock final") :]
    instance_state = instance_state[: instance_state.index("}; }")]
    assert "tracking_mutex" in instance_state
    assert (
        "std::unordered_multimap<std::uint64_t, InstanceControlBlock*> registry;"
        in instance_state
    )
    buffer_state = first[first.index("struct BufferControlBlock final") :]
    buffer_state = buffer_state[: buffer_state.index("}; }")]
    assert "tracking_mutex" in buffer_state
    assert "registry;" not in buffer_state
    physical_impl = first[
        first.index("inline Result<PhysicalDevice> PhysicalDevice::borrow") :
    ]
    physical_impl = physical_impl[
        : physical_impl.index("inline Result<PhysicalDevice> PhysicalDevice::adopt")
    ]
    assert "registry.equal_range(detail::raw_key(native))" in physical_impl
    assert "detail::same_object(found->second->parent, parent)" in physical_impl
    assert (
        "if constexpr (requires { left.parent(); }) return same_object(left.parent(), right.parent());"
        in first
    )
    assert "const Device& parent() const noexcept" in first
    assert (
        "detail::DispatchState{parent().dispatchState().instance, ctrl_ ? &ctrl_->device_dispatch : nullptr, native_}"
        in first
    )
    assert "detail::DispatchState{parent_.dispatchState().instance" not in first
    assert "auto value = Buffer(native, parent); return value;" in first
    assert "borrow_state" not in first
    fence = first[first.index("class Fence") :]
    fence = fence[: fence.index("};")]
    assert "createInfo()" not in fence
    assert "class Buffer" in first
    assert "useBufferEXT" in first
    assert (
        "ResultValue<std::vector<Buffer>> enumerateBuffersEXT(std::uint32_t bufferCount = 0) const;"
        in first
    )
    assert "Device::enumerateBuffersEXT(std::uint32_t bufferCount) const {" in first
    assert (
        "Result<void> createGraphicsPipelines(const PipelineCache& pipelineCache, std::span<const GraphicsPipelineCreateInfo> createInfos, std::optional<std::reference_wrapper<const AllocationCallbacks>> allocator, std::span<Pipeline> pipelines) const;"
        in first
    )
    assert "createGraphicsPipelines(VkPipelineCache" not in first
    assert "createGraphicsPipelines" in first
    assert "createInfoCount = 0" not in first
    queue = first[first.index("class Queue {") :]
    queue = queue[: queue.index("\n};")]
    assert (
        "Result<void> submit(std::span<const SubmitInfo> submits, const Fence& fence) const;"
        in queue
    )
    assert "queueSubmit(" not in queue
    device = first[first.index("class Device {") :]
    device = device[: device.index("\n};")]
    assert "Result<void> waitIdle() const;" in device
    assert "deviceWaitIdle(" not in device
    assert "cmdBindPipeline(" not in first
    assert "std::span<const std::span" not in first
    assert (
        "Result<void> buildAccelerationStructuresKHR(std::span<const AccelerationStructureBuildGeometryInfoKHR> infos, std::span<const AccelerationStructureBuildRangeInfoKHR> buildRangeInfos) const;"
        in first
    )
    build = first[
        first.index("inline Result<void> Device::buildAccelerationStructuresKHR") :
    ]
    build = build[: build.index("\n}")]
    assert (
        "for (const auto& info : infos_native) buildRangeInfos_required += info.geometryCount;"
        in build
    )
    assert "buildRangeInfos.size() != buildRangeInfos_required" in build
    geometry_info = first[
        first.index("struct AccelerationStructureBuildGeometryInfoKHR {") :
    ]
    geometry_info = geometry_info[: geometry_info.index("\n};")]
    assert (
        "std::vector<AccelerationStructureGeometryKHR> geometries{};" in geometry_info
    )
    assert "std::vector<std::vector" not in geometry_info
    geometry_conversion = first[
        first.index(
            "inline void AccelerationStructureBuildGeometryInfoKHR::to_cstruct"
        ) :
    ]
    geometry_conversion = geometry_conversion[: geometry_conversion.index("\n}")]
    assert (
        "ppGeometries_pointers[i] = &output->ppGeometries_native[i];"
        in geometry_conversion
    )
    profile_list = first[first.index("struct VideoProfileListInfoKHR {") :]
    profile_list = profile_list[: profile_list.index("\n};")]
    assert "std::vector<VideoProfileInfoKHR> profiles{};" in profile_list
    assert "uint32_t profileCount{};" in profile_list
    assert "setProfileCount(uint32_t value)" in profile_list
    assert (
        "setProfiles(std::vector<VideoProfileInfoKHR> value) & { profiles = std::move(value); profileCount = static_cast<uint32_t>(profiles.size()); return *this; }"
        in profile_list
    )
    assert "pProfiles{};" not in profile_list
    assert "setPProfiles" not in profile_list
    conversion = first[
        first.index("inline void VideoProfileListInfoKHR::from_cstruct") :
    ]
    conversion = conversion[: conversion.index("\n}")]
    assert (
        "profiles.resize(static_cast<std::size_t>(native.profileCount));" in conversion
    )
    assert "profiles[i].from_cstruct(native.pProfiles[i]);" in conversion
    assert (
        "mutable std::shared_mutex externsync"
        not in first[
            first.index("struct BufferControlBlock final") : first.index(
                "}; }", first.index("struct BufferControlBlock final")
            )
        ]
    )
    assert "mutable std::shared_mutex externsync" in lifetime
    assert "collect(value.parent()" not in first
    assert (
        "reinterpret_cast<std::uintptr_t>(static_cast<detail::LifetimeHeader*>(state))"
        in first
    )
    assert "return std::unexpected(ResultCode::ErrorOutOfHostMemory);" in first
    assert "ctrl_->data->insert_or_assign(typeid(T), std::move(value))" in first
    context = first[
        first.index("class Context") : first.index("};", first.index("class Context"))
    ]
    assert "static Result<Context> create()" in context
    assert "volkGetInstanceVersion() < minimumApiVersion" in first
    assert (
        "std::unexpected(static_cast<ResultCode>(VK_ERROR_INCOMPATIBLE_DRIVER))"
        in first
    )
    assert "ResultValue<std::vector<Pipeline>> createGraphicsPipelines(" in first
    assert "Result<Buffer> createBuffer(" in first
    assert "ResultValue<Buffer> createBuffer(" not in first
    create_buffer_start = first.index("inline Result<Buffer> Device::createBuffer")
    create_buffer_end = first.index(
        "inline ResultValue<std::vector<Buffer>> Device::enumerateBuffersEXT",
        create_buffer_start,
    )
    create_buffer = first[create_buffer_start:create_buffer_end]
    assert "return std::unexpected(status);" in create_buffer
    assert "return value;" in create_buffer
    assert "this->dispatchState().device->vkCreateBuffer" in first
    create_buffer_impl = first[
        first.index("inline Result<void> Device::createBuffer") :
    ]
    create_buffer_impl = create_buffer_impl[
        : create_buffer_impl.index("inline Result<Buffer> Device::createBuffer")
    ]
    assert "](const Device& owner, VkBuffer value)" in create_buffer_impl
    assert "release_device = *this" not in create_buffer_impl
    assert run(arguments + ["--check"]) == 0
    assert output.read_text(encoding="utf-8") == first


def test_check_reports_changed_output(tmp_path):
    output = tmp_path / "wrapper.hpp"
    output.write_text("stale", encoding="utf-8")
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(output),
                "--check",
            ]
        )
        == 1
    )
    assert output.read_text(encoding="utf-8") == "stale"


def test_duplicate_output_is_rejected(tmp_path):
    from vulkan_wrapper_gen.template import TemplateError
    import pytest

    output = tmp_path / "same.hpp"
    with pytest.raises(TemplateError, match="duplicate output"):
        run(
            [
                "--registry",
                str(FIXTURE),
                "--emit", str(ROOT / "templates" / "vulkan.template.hpp") + ":" + str(output),
                "--emit", str(ROOT / "templates" / "vulkan.template.cpp") + ":" + str(output),
            ]
        )


def test_unknown_config_receiver_is_rejected(tmp_path):
    from vulkan_wrapper_gen.config import ConfigError
    import pytest

    config = tmp_path / "bad.toml"
    config.write_text(
        'version = 1\n[receivers.vkMissing]\nadd = ["VkBuffer"]\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="unknown command vkMissing"):
        run(
            [
                "--registry",
                str(FIXTURE),
                "--config",
                str(config),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(tmp_path / "wrapper.hpp"),
            ]
        )


def test_paired_templates_separate_declarations_and_implementations(tmp_path):
    header = tmp_path / "vulkan_wrapper.hpp"
    source = tmp_path / "vulkan_wrapper.cpp"
    assert (
        run(
            [
                "--registry",
                str(FIXTURE),
                "--emit", str(ROOT / "templates" / "vulkan.template.hpp") + ":" + str(header),
                "--emit", str(ROOT / "templates" / "vulkan.template.cpp") + ":" + str(source),
            ]
        )
        == 0
    )
    declarations = header.read_text(encoding="utf-8")
    implementations = source.read_text(encoding="utf-8")
    assert declarations.index("class Device {") < declarations.index("class Context {")
    assert declarations.index("class Context {") < declarations.index(
        "struct BufferCreateInfo {"
    )
    assert "Result<void> createBuffer(" in declarations
    assert "Device::createBuffer(" not in declarations
    assert "void BufferCreateInfo::to_cstruct" not in declarations
    assert "template <typename T> inline Result<void> Buffer::setUserData" in declarations
    assert "void Buffer::useEXT(" in implementations
    assert "void BufferCreateInfo::to_cstruct" in implementations
    assert (
        "template <typename T> inline Result<void> Buffer::setUserData"
        not in implementations
    )
