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


def test_shared_counts_never_exceed_owned_array_storage(tmp_path):
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
    parallel = generated[generated.index("inline void ParallelArrays::to_cstruct") :]
    parallel = parallel[: parallel.index("\n}")]
    assert "std::size_t capacity = left.size();" in parallel
    assert "if (candidate < capacity) capacity = candidate;" in parallel
    alternative = generated[
        generated.index("inline void AlternativeArrays::to_cstruct") :
    ]
    alternative = alternative[: alternative.index("\n}")]
    assert "std::size_t capacity{};" in alternative
    assert "candidate != 0 && (capacity == 0 || candidate < capacity)" in alternative


def test_non_derived_count_member_stays_explicit(tmp_path):
    # descriptorCount is the number of descriptors in a binding, not the length
    # of pImmutableSamplers (which is only conditionally that long), so it must
    # stay an explicit field; viewportCount, in contrast, is a plain array
    # length and stays derived.
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
    assert "setDescriptorCount" in binding
    binding_impl = generated[generated.index("BindingLike::to_cstruct") :]
    binding_impl = binding_impl[: binding_impl.index("\n}")]
    assert "descriptorCount = descriptorCount;" in binding_impl
    viewport = generated[generated.index("struct ViewportLike {") :]
    viewport = viewport[: viewport.index("\n};")]
    assert "viewportCount" not in viewport
    viewport_impl = generated[generated.index("ViewportLike::to_cstruct") :]
    viewport_impl = viewport_impl[: viewport_impl.index("\n}")]
    assert "viewportCount = static_cast<uint32_t>(viewports.size());" in viewport_impl


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
    assert "size_t codeSize" not in shader
    assert "std::vector<uint32_t> code{};" in shader
    assert "setCodeSize" not in shader
    assert "output->value.codeSize = static_cast<size_t>(code.size() * 4);" in generated
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
    assert "setProfiles(std::vector<VideoProfileInfoKHR> value)" in profile_list
    assert "uint32_t profileCount{};" not in profile_list
    assert "pProfiles{};" not in profile_list
    assert "setProfileCount" not in profile_list
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
    assert "template <typename T> inline Result<void> Buffer::setData" in declarations
    assert "void Device::useBufferEXT(" in implementations
    assert "void BufferCreateInfo::to_cstruct" in implementations
    assert (
        "template <typename T> inline Result<void> Buffer::setData"
        not in implementations
    )
