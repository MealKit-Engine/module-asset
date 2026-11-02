#pragma once

#include "mealkit/core/Types.h"

#include <string>

// Asset budgets (docs/ROADMAP.md stage 2 "Asset budget warnings", docs/CODING_MODEL.md "Model budget kinds"). Model kind is decided by folder:
//   assets/models/characters/... = character, models/environment/... = environment, anything else = prop
// Limits are in mealkit.project (budget<kind><item>), defaults in the table below (based on min-spec integrated graphics, tuned by benchmark).
// Exceeding them warns on load, and the editor inspector and Blender add-on show the same values. Unrelated to the simulation.
namespace mealkit::asset {

enum class ModelKind : u8 { Character, Prop, Environment, Count };
inline constexpr u32 kModelKindCount = static_cast<u32>(ModelKind::Count);

struct ModelBudget {
    u32 triangles = 0;   // LOD 0
    u32 textureSize = 0; // longest side (pixels)
    u32 bones = 0;
    u32 materials = 0;
    // Not a load warning: instances smaller on screen than this (bounding radius / distance, the LOD measure) aren't drawn. 0 = always drawn
    f32 cullSize = 0.0f;
};

inline constexpr ModelBudget kDefaultBudgets[kModelKindCount] = {
    {15000, 2048, 64, 4}, // character
    {3000, 1024, 0, 2, 0.004f}, // prop (0.004 = about 2.5 px radius at 720p)
    {20000, 2048, 0, 4},  // environment
};

// Folder name = kind name ("characters", "props", "environment")
inline const char* modelKindFolder(ModelKind kind) {
    switch (kind) {
    case ModelKind::Character: return "characters";
    case ModelKind::Environment: return "environment";
    default: return "props";
    }
}

// Kind of a model path relative to assets ("models/characters/hero.glb"). Prop if not in a kind folder
inline ModelKind modelKind(const std::string& path) {
    std::string p = path;
    for (char& c : p)
        if (c == '\\') c = '/';
    const std::string prefix = "models/";
    const size_t start = p.rfind(prefix, 0) == 0 ? prefix.size() : 0;
    const size_t slash = p.find('/', start);
    if (slash == std::string::npos) return ModelKind::Prop;
    const std::string folder = p.substr(start, slash - start);
    for (u32 k = 0; k < kModelKindCount; ++k)
        if (folder == modelKindFolder(static_cast<ModelKind>(k))) return static_cast<ModelKind>(k);
    return ModelKind::Prop;
}

} // namespace mealkit::asset
