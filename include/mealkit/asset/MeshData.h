#pragma once

#include "mealkit/core/Math.h"

#include <vector>

// CPU mesh, material and texture data: what model import, cooking and terrain produce and the renderer uploads.
// The renderer (render/Mesh.h, render/Renderer.h) uses these same types; asset never includes render or rhi
// (docs/ENGINE_MODULES.md "Modules"). Layouts here are part of the cooked model format (asset/Cooked.h kLayout).
namespace mealkit::asset {

// 36 bytes (docs/DECISIONS.md 2026-10-21 "Vertex format"): texture coordinates are two halfs, material is an index into the renderer's material table
struct Vertex {
    Vec3 position;
    Vec3 normal;
    u32 color = 0xFFFFFFFF; // RGBA8 sRGB. Multiplied with the instance color and material base color (glTF COLOR_0)
    u32 uv = 0;             // TEXCOORD_0 (half u | half v << 16, packHalf2)
    u16 material = 0;       // Renderer::createMaterial index (0 = default white)
    u16 reserved = 0;
};
static_assert(sizeof(Vertex) == 36);

// two floats → two halfs (rounded, out of range becomes infinity)
u32 packHalf2(f32 u, f32 v);

// Four joints of a skinned vertex (skeletal animation): indices u8 ×4, weights unorm8 ×4 (sum 255). On the GPU appended after Vertex for 44 bytes
struct SkinWeights {
    u32 joints = 0;
    u32 weights = 0xFFu; // first joint 1
};
static_assert(sizeof(SkinWeights) == 8);
inline constexpr u32 kSkinnedVertexStride = sizeof(Vertex) + sizeof(SkinWeights);

// One slot of the skin palette (48 bytes read by the HLSL vertex shader: three rows, p' = (dot(r0, p1), dot(r1, p1), dot(r2, p1))). Filled by asset::skinPalette
struct BoneMatrix {
    f32 rows[3][4];
};
static_assert(sizeof(BoneMatrix) == 48);

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<u32> indices; // with fewer than 65536 vertices, uploaded to the GPU as 16-bit
    std::vector<SkinWeights> skin; // empty for static meshes, one per vertex if present (skinned mesh)
};

// Box centered at the origin with side 1. Each face has its own vertices so normals come out faceted.
MeshData makeBox();

// Renderer mesh handle (Renderer::createMesh)
using MeshId = u32;
inline constexpr MeshId kNoMesh = 0xFFFFFFFFu; // not drawn (ModelCache::pickLod below the cull size)

// Materials (supported range of glTF PBR metallic-roughness, docs/CODING_MODEL.md "Models, materials, textures"). The renderer keeps them in a table
// and writes it to the upload ring every frame (shaders/Forward.hlsl Material, 64 bytes). Textures are bindless indices (kNoTexture = none)
inline constexpr u32 kNoTexture = 0xFFFFFFFFu;

// Material features (glTF extras "mealkit_<key>", materialFeatures in asset::modelSchemaJson). Not shader variants but
// chosen inside the shader by material table bits: even if materials with different features mix in one mesh, it's still one draw call
inline constexpr u32 kMaterialOutline = 1u;  // outline: inverted hull pass (if any material in a mesh has it, that mesh is drawn once more)
inline constexpr u32 kMaterialWind = 2u;     // wind sway: strength is painted in vertex color alpha (1 for models without alpha)
inline constexpr u32 kMaterialRim = 4u;      // rim light: edges get brighter
inline constexpr u32 kMaterialDissolve = 8u; // dissolve: disappears by the instance's dissolve progress (instanceDissolve)

struct MaterialData {
    f32 baseColor[4] = {1, 1, 1, 1}; // linear
    f32 emissive[3] = {0, 0, 0};
    f32 metallic = 0.0f;
    f32 roughness = 1.0f;
    f32 normalScale = 1.0f;
    f32 alphaCutoff = 0.0f; // > 0 means alpha mask (alpha below this isn't drawn)
    u32 baseColorTexture = kNoTexture;
    u32 normalTexture = kNoTexture;
    u32 metallicRoughnessTexture = kNoTexture; // G = roughness, B = metal (glTF)
    u32 emissiveTexture = kNoTexture;
    // Style preset material features (kMaterial* bits and values). Colors are linear
    u32 features = 0;
    f32 outlineWidth = 0.02f; // meters (world units, independent of instance scale)
    f32 outlineColor[3] = {0, 0, 0};
    f32 windStrength = 0.1f;  // meters (sway amplitude at vertex alpha 1 and FrameParams::windStrength 1)
    f32 rimStrength = 0.5f;
    f32 rimColor[3] = {1, 1, 1};
    f32 dissolveEdge = 0.05f; // glowing width of the disappearing edge (in progress units 0..1)
    f32 dissolveColor[3] = {4.0f, 1.6f, 0.4f}; // edge emission (linear, brighter above 1)
};

// Model textures (render/TextureCache.h compresses them): what a glTF slot is used for
enum class TextureUsage : u8 {
    Color,  // baseColor·emissive: sRGB BC7
    Linear, // metallicRoughness: BC7 (linear)
    Normal, // normal map: BC5 (x, y. The shader reconstructs z)
};

// Block-compressed texture formats. Same values as rhi::Format (render checks it), so cooked models store them unchanged
enum class TextureFormat : u8 { Bc7Unorm = 3, Bc7Srgb = 4, Bc5Unorm = 5 };

struct CompressedTexture {
    u32 width = 0, height = 0, mipLevels = 0;
    TextureFormat format = TextureFormat::Bc7Unorm;
    std::vector<u8> data; // from mip 0, 16 bytes per 4x4 block each (rhi::textureMipBytes, rhi::TextureInit)
};

} // namespace mealkit::asset
