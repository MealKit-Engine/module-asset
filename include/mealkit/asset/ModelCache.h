#pragma once

#include "mealkit/asset/Animation.h"
#include "mealkit/asset/Budget.h"
#include "mealkit/asset/MeshData.h"
#include "mealkit/asset/ModelGpu.h"
#include "mealkit/asset/Scatter.h"
#include "mealkit/asset/Terrain.h"
#include "mealkit/core/Math.h"
#include "mealkit/scene/Scene.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// glTF (.glb/.gltf) model loading and cache. Shipping builds read cooked models (asset/Cooked.h) from the pack instead.
// On import, meshoptimizer dedups vertices, optimizes for cache/overdraw/fetch, and makes up to 2 extra LOD levels.
// Convex hull points for physics (hullPoints) are taken straight from the file's vertices, so they're the same on every PC.
// Drawing is the render module's (render/ModelDraw.h: addModel, addTerrain, addScatter, postParams, probes); GPU uploads go through ModelGpu.
namespace mealkit::asset {

inline constexpr u32 kMaxLods = 3;

struct Model {
    bool valid = false;
    std::string path; // path as written in the scene (relative to assets)
    MeshId lods[kMaxLods] = {};
    u32 lodCount = 0;
    u32 triangles[kMaxLods] = {};
    Vec3 boundsMin, boundsMax;    // model space (MealKit coordinates: left-handed, y up)
    std::vector<Vec3> hullPoints; // for the physics convex hull
    u64 fileHash = 0;             // hash of the original file content (for multiplayer checks)
    // Asset budget (asset/Budget.h): kind from the path's folder, usage counted on load. Exceeded items go in budgetProblems (empty = within budget)
    ModelKind kind = ModelKind::Prop;
    u32 materials = 0;
    u32 bones = 0;       // joint count of the skin (over kMaxJoints, the skin is dropped and it becomes a static model)
    // Skeletal animation (asset/Animation.h): with a skin, lods[0] alone is the skinned mesh (no LODs), clips are the glTF animations
    Skeleton skeleton;
    std::vector<AnimationClip> clips;
    bool skinned() const { return skeleton.valid(); }
    u32 textureSize = 0; // longest side of the largest texture (when there is a renderer)
    u32 textures = 0;
    std::vector<std::string> budgetProblems;
};

// Heightmap of a terrain object (asset/Terrain.h) + its view chunk meshes (kTerrainLods per chunk, built on first draw and after edits)
struct Terrain {
    std::string path; // relative to assets
    Heightmap map;
    unsigned long long fileTime = 0;
    // View
    Vec3 meshSize;                      // box size the meshes were built for (rebuilt when it changes)
    u32 chunks = 0;                     // per side
    std::vector<MeshId> meshes; // (cz * chunks + cx) * kTerrainLods + lod, kNoMesh until built
    std::vector<u8> stale;              // per chunk: rebuild before drawing
};

// Items where the model exceeds its kind's budget ("15200 triangles (budget 15000)"). English (log/inspector)
std::vector<std::string> budgetProblems(const Model& model, const ModelBudget& budget);

// Schema (JSON) telling external tools the supported model range. Read by the Blender add-on (tools/blender-addon/mealkit/schema.json),
// and the asset_schema unit test checks that file matches this (so the engine and add-on don't drift). Update together when loading changes
std::string modelSchemaJson();

// "mealkit_<key>" in glTF material extras (JSON object) → style preset material features (schema materialFeatures, written by the Blender add-on
// as custom properties). Main feature keys (outline, wind, rim, dissolve) turn the feature on; color keys only change that feature's values.
// Unknown mealkit_ keys and bad values go to problems in English (loading warns with the file name). Keys not starting with mealkit_ are ignored
void applyMaterialExtras(const char* json, MaterialData& material, std::vector<std::string>& problems);

class ModelCache {
public:
    // Without one (headless), only CPU data (bounds, hull, hash) is made. render::ModelUploader is the renderer's
    void setGpu(ModelGpu* gpu) { m_gpu = gpu; }
    bool hasGpu() const { return m_gpu != nullptr; }

    // Loaded on first request and served from the cache after that. nullptr on failure (the same path isn't retried).
    const Model* get(const std::string& path);
    MeshId boxMesh(); // default box with side 1
    // Sky HDRI (scene environment hdri, path relative to assets) → renderer environment index (FrameParams::environment). On first request, reads from the cache or
    // bakes (dev build), afterwards only returns the index. 0 if empty, no renderer, or failure (gradient; warns once per path).
    // When reloadChanged sees the file changed, it rebakes on the next request (old GPU textures remain)
    u32 environment(const std::string& hdriPath);
    // Reflection probe file (.mkprobe relative to assets) → renderer probe index. Like environment, uploaded on first request and re-uploaded when changed
    u32 probe(const std::string& probePath);
    // Color grading LUT (scene environment post.lut, .cube relative to assets) → renderer LUT index (PostParams::lut). Like environment, uploaded on first request,
    // and re-uploaded when changed. 0 if empty or failed (no grading)
    u32 lut(const std::string& lutPath);
    // Probes per scene (project setting probeMaxCount; render::probes also clamps to the renderer's limit)
    void setProbeLimit(u32 count) { m_probeLimit = std::max<u32>(count, 1); }
    u32 probeLimit() const { return m_probeLimit; }

    // Asset hot reload (editor edit mode): rereads model files that were ever requested and have changed (including files that newly appeared).
    // Model addresses stay; only contents change. Returns the list of reloaded paths. On failure, keeps the previous contents.
    // Old GPU meshes aren't deleted (the renderer has no mesh deletion; they only pile up during development)
    std::vector<std::string> reloadChanged();

    // Per-kind budgets (mealkit.project, game::modelBudgets). Already loaded models are rechecked too
    void setBudgets(const ModelBudget (&budgets)[kModelKindCount]);
    const ModelBudget& budget(ModelKind kind) const { return m_budgets[static_cast<u32>(kind)]; }

    // Picks the LOD by on-screen size. kNoMesh below the kind's cullSize (DrawList::add skips it)
    MeshId pickLod(const Model& model, Vec3 cameraPos, Vec3 position, Vec3 scale) const;

    // Terrain heightmap (path relative to assets): loaded on first request, nullptr if missing/unreadable (warns once).
    // Painting is MkRender's; a changed file is reread by reloadChanged
    Terrain* terrain(const std::string& path);
    // Builds the terrain's stale chunk meshes for this box size (first use, edits, a new size). false without a GPU or heightmap.
    // render::addTerrain calls it outside a frame and then draws terrain.meshes
    bool buildTerrainMeshes(Terrain& terrain, Vec3 size);

    // Scatter layer instances (.scatter relative to assets): loaded on first request; a missing file is an empty layer (painted in MkRender)
    struct ScatterEntry {
        Scatter scatter;
    };
    ScatterEntry& scatter(const std::string& path);
    // Hash of the simulation data of the models the scene references: physics hull points and terrain heightmaps
    // (combined with scene.contentHash() for connection checks; visual meshes, textures and materials aren't included, so platform cooking doesn't change it).
    u64 hashModels(const scene::Scene& scene);

    static std::string resolve(const std::string& path); // path relative to assets → real file path
    static std::string assetsRoot();                     // assets folder near the repo/executable

private:
    bool load(const std::string& path, Model& out);
    std::unordered_map<std::string, std::unique_ptr<ScatterEntry>> m_scatters;
    std::unordered_map<std::string, std::unique_ptr<Terrain>> m_terrains; // nullptr = failed (retried when the file changes)
    ModelGpu* m_gpu = nullptr;
    struct EnvironmentEntry {
        u32 id = 0; // 0 = failed (not retried)
        unsigned long long time = 0;
    };
    std::unordered_map<std::string, EnvironmentEntry> m_environments;
    std::unordered_map<std::string, EnvironmentEntry> m_probes;
    std::unordered_map<std::string, EnvironmentEntry> m_luts;
    u32 m_probeLimit = 8;
    MeshId m_box = 0xFFFFFFFFu;
    std::unordered_map<std::string, std::unique_ptr<Model>> m_models;
    std::unordered_map<std::string, unsigned long long> m_fileTimes;
    ModelBudget m_budgets[kModelKindCount] = {kDefaultBudgets[0], kDefaultBudgets[1], kDefaultBudgets[2]};
    void checkBudget(Model& model) const; // file time at load (reloadChanged)
};

} // namespace mealkit::asset
