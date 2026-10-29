#pragma once

#include "mealkit/rhi/Rhi.h"

#include <string>
#include <vector>

// Model texture cache (docs/ROADMAP.md stage 2 "Textures and materials", docs/DECISIONS.md 2026-10-21 "Texture cache").
// PNG/JPEG inside glTF is compressed with mipmaps to BC7 (color/metal-roughness/emissive) or BC5 (normals) and saved to a file:
//   <project>/build/cache/textures/<content hash>.mktex   (outside git, regenerated if deleted)
// Compression only happens in dev builds (builds with debug UI: editor/dev exe). Shipping builds only read the compressed cache
// (Export copies it to cache/textures next to the exe). View-only, so unrelated to the simulation and content hash
// (the source images are already in the glTF file hash).
namespace mealkit::asset {

enum class TextureUsage : u8 {
    Color,  // baseColor·emissive: sRGB BC7
    Linear, // metallicRoughness: BC7 (linear)
    Normal, // normal map: BC5 (x, y. The shader reconstructs z)
};

struct CompressedTexture {
    u32 width = 0, height = 0, mipLevels = 0;
    rhi::Format format = rhi::Format::Bc7Unorm;
    std::vector<u8> data; // from mip 0, rhi::textureMipBytes each (rhi::TextureInit)
};

// Encoded image (PNG/JPEG bytes in glTF) → read from the cache, or if missing decode, mipmap, compress, and write to cache. On failure, false and a reason
bool loadCompressedTexture(const void* encoded, size_t bytes, TextureUsage usage, CompressedTexture& out, std::string& error);

// Cache folder: written to <project>/build/cache/textures; when reading, also cache/textures next to the exe (shipping)
std::string textureCacheFolder();

// Mipmaps only without compression (for unit tests, RGBA8). With srgb, converts to linear before averaging
void buildMips(const u8* rgba, u32 width, u32 height, bool srgb, bool normal, std::vector<std::vector<u8>>& mips);

} // namespace mealkit::asset
