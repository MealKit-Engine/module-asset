#pragma once

#include "mealkit/asset/Animation.h"
#include "mealkit/asset/Cutscene.h"
#include "mealkit/asset/MeshData.h"
#include "mealkit/asset/ModelGpu.h"

#include <string>
#include <string_view>
#include <vector>

// Cooked assets (docs/ROADMAP.md stage 4 "Asset cooking + pack files"): engine binaries the cook step (game/Cook.h) writes into the pack
// next to where the source was ("models/crate.glb" → "models/crate.glb.mkmodel"). Loading them needs no parsing, mesh optimization or
// texture compression, so shipping builds carry no glTF reader (cgltf) or meshoptimizer: those are only in builds with MEALKIT_ASSET_IMPORT
// (dev builds, the editor, headless). Readers check the source first and the cooked file second, so the editor keeps using the sources.
// Little-endian blobs with a magic and a version: a version mismatch is "cook again", never a migrate (the sources are the truth).
namespace mealkit::asset {

inline constexpr const char* kCookedModelSuffix = ".mkmodel";
inline constexpr const char* kCookedCameraSuffix = ".mkcam";
inline constexpr const char* kCookedEnvironmentSuffix = ".mkenv";

// Everything ModelCache needs from a model file, before anything goes to the GPU
struct CookedModel {
    u64 fileHash = 0; // hash of the source file (content hash: the same in the editor and the shipping game)
    Vec3 boundsMin, boundsMax;
    std::vector<Vec3> hullPoints; // from the source vertices before optimization (determinism)
    u32 bones = 0;                // skin joint count, also when the skin was dropped (budget)
    Skeleton skeleton;
    std::vector<AnimationClip> clips;
    // Texture fields of materials are indices into textures (kNoTexture = none). Vertex material = index into materials
    std::vector<MaterialData> materials;
    std::vector<CompressedTexture> textures;
    std::vector<MeshData> lods; // lods[0] full detail
};

std::string writeCookedModel(const CookedModel& model);
bool readCookedModel(std::string_view bytes, CookedModel& out, std::string& error);

std::string writeCameraClips(const std::vector<CameraClip>& clips);
bool readCameraClips(std::string_view bytes, std::vector<CameraClip>& out, std::string& error);

#if MEALKIT_ASSET_IMPORT
// compress: asset/ModelGpu.h TextureCompressor (nullptr = materials without textures, headless)
// glTF file (absolute path) → cooked model: vertices deduplicated and optimized, up to 2 extra LODs (asset/ModelImport.cpp)
bool importModel(const std::string& file, CookedModel& out, TextureCompressor compress, std::string& error);
#endif

} // namespace mealkit::asset
