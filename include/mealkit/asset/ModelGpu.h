#pragma once

#include "mealkit/asset/MeshData.h"

#include <cstddef>
#include <string>

// What ModelCache hands to the GPU. The render module implements it (render::ModelUploader in render/ModelDraw.h), so asset never
// includes render or rhi; headless builds set none and ModelCache makes only CPU data (bounds, hulls, hashes).
namespace mealkit::asset {

// Texture compression for model import (render::loadCompressedTexture, front end). nullptr = materials without textures
using TextureCompressor = bool (*)(const void* encoded, size_t bytes, TextureUsage usage, CompressedTexture& out, std::string& error);

class ModelGpu {
public:
    virtual ~ModelGpu() = default;
    virtual MeshId createMesh(const MeshData& mesh) = 0;
    virtual void replaceMesh(MeshId id, const MeshData& mesh) = 0;
    virtual u32 createMaterial(const MaterialData& material) = 0;
    virtual u32 createTexture(const CompressedTexture& texture) = 0; // kNoTexture on failure
    virtual TextureCompressor textureCompressor() const = 0;
    // Files resolved under assets/ → renderer indices. On failure, 0 and a reason
    virtual u32 createEnvironment(const std::string& file, std::string& error) = 0; // sky HDRI (.hdr or cooked .mkenv)
    virtual u32 createProbe(const std::string& file, std::string& error) = 0;       // baked reflection probe (.mkprobe)
    virtual u32 createLut(const std::string& file, std::string& error) = 0;         // color grading LUT (.cube)
};

} // namespace mealkit::asset
