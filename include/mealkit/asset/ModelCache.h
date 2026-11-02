#pragma once

#include "mealkit/asset/Animation.h"
#include "mealkit/asset/Budget.h"
#include "mealkit/asset/Environment.h"
#include "mealkit/asset/TextureCache.h"
#include "mealkit/asset/Scatter.h"
#include "mealkit/asset/Terrain.h"
#include "mealkit/core/Math.h"
#include "mealkit/render/DrawList.h"
#include "mealkit/render/Renderer.h"
#include "mealkit/scene/Scene.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// glTF (.glb/.gltf) model loading and cache. Shipping builds read cooked models (asset/Cooked.h) from the pack instead.
// On import, meshoptimizer dedups vertices, optimizes for cache/overdraw/fetch, and makes up to 2 extra LOD levels.
// Convex hull points for physics (hullPoints) are taken straight from the file's vertices, so they're the same on every PC.
namespace mealkit::asset {

inline constexpr u32 kMaxLods = 3;

struct Model {
    bool valid = false;
    std::string path; // path as written in the scene (relative to assets)
    render::MeshId lods[kMaxLods] = {};
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
    bool dirty = false; // edited in the editor and not written to the file yet
    unsigned long long fileTime = 0;
    // View
    Vec3 meshSize;                      // box size the meshes were built for (rebuilt when it changes)
    u32 chunks = 0;                     // per side
    std::vector<render::MeshId> meshes; // (cz * chunks + cx) * kTerrainLods + lod, kNoMesh until built
    std::vector<u8> stale;              // per chunk: rebuild before drawing
};
inline constexpr render::MeshId kNoMesh = 0xFFFFFFFFu;

// Adds one model to the list: static models via add, skinned models via addSkinned with a palette looping anim ("<clip> [seconds offset]", the scene's kAnimField)
// at seconds (rest pose if no or empty clip). mesh is the pickLod result (lods[0] for skinned models)
void addModel(render::DrawList& out, const Model& model, render::MeshId mesh, const render::InstanceData& instance,
              const std::string& anim = {}, f64 seconds = 0.0);

// Items where the model exceeds its kind's budget ("15200 triangles (budget 15000)"). English (log/inspector)
std::vector<std::string> budgetProblems(const Model& model, const ModelBudget& budget);

// Schema (JSON) telling external tools the supported model range. Read by the Blender add-on (tools/blender-addon/mealkit/schema.json),
// and the asset_schema unit test checks that file matches this (so the engine and add-on don't drift). Update together when loading changes
std::string modelSchemaJson();

// "mealkit_<key>" in glTF material extras (JSON object) → style preset material features (schema materialFeatures, written by the Blender add-on
// as custom properties). Main feature keys (outline, wind, rim, dissolve) turn the feature on; color keys only change that feature's values.
// Unknown mealkit_ keys and bad values go to problems in English (loading warns with the file name). Keys not starting with mealkit_ are ignored
void applyMaterialExtras(const char* json, render::MaterialData& material, std::vector<std::string>& problems);

class ModelCache {
public:
    // Without a renderer, only CPU data (bounds, hull, hash) is made (headless).
    // Renderer calls are only inside this inline function, so headless builds that never call it link without the renderer.
    void setRenderer(render::Renderer* renderer) {
        m_uploadContext = renderer;
        m_upload = renderer ? [](void* r, const render::MeshData& mesh) {
            return static_cast<render::Renderer*>(r)->createMesh(mesh);
        } : nullptr;
        m_replace = renderer ? [](void* r, render::MeshId id, const render::MeshData& mesh) {
            static_cast<render::Renderer*>(r)->replaceMesh(id, mesh);
        } : nullptr;
        m_uploadMaterial = renderer ? [](void* r, const render::MaterialData& m) {
            return static_cast<render::Renderer*>(r)->createMaterial(m);
        } : nullptr;
        // Textures: compressed by the import (read from the texture cache or compressed, asset/TextureCache.cpp, front end), uploaded here
        m_compressTexture = renderer ? &loadCompressedTexture : nullptr;
        m_uploadTexture = renderer ? [](void* r, const CompressedTexture& t) {
            return static_cast<render::Renderer*>(r)->createTexture({t.width, t.height, t.mipLevels, t.format, t.data.data()});
        } : nullptr;
        // Sky HDRI: read from the cache or bake and upload (asset/Environment.cpp, front end). On failure, 0 and a reason
        m_uploadEnvironment = renderer ? [](void* r, const std::string& file, std::string& error) -> u32 {
            EnvironmentMap e;
            if (!loadEnvironment(file, e, error)) return 0;
            render::EnvironmentData d;
            std::memcpy(d.sh, e.sh, sizeof(d.sh));
            d.skySize = e.skySize;
            d.sky = e.sky.data();
            d.reflectSize = e.reflectSize;
            d.reflectMips = e.reflectMips;
            d.reflect = e.reflect.data();
            return static_cast<render::Renderer*>(r)->createEnvironment(d);
        } : nullptr;
        // Reflection probes: upload the baked files from assets/probes/. On failure, 0 and a reason
        m_uploadProbe = renderer ? [](void* r, const std::string& file, std::string& error) -> u32 {
            ProbeMap probe;
            if (!loadProbe(file, probe, error)) return 0;
            return static_cast<render::Renderer*>(r)->createProbe(probe.size, probe.mips, probe.reflect.data());
        } : nullptr;
        // Color grading LUT: read the .cube and upload. On failure, 0 and a reason
        m_uploadLut = renderer ? [](void* r, const std::string& file, std::string& error) -> u32 {
            std::vector<f32> rgb;
            if (!loadCubeLut(file, rgb, error)) return 0;
            return static_cast<render::Renderer*>(r)->createLut(rgb.data());
        } : nullptr;
    }

    // Loaded on first request and served from the cache after that. nullptr on failure (the same path isn't retried).
    const Model* get(const std::string& path);
    render::MeshId boxMesh(); // default box with side 1
    // Sky HDRI (scene environment hdri, path relative to assets) → renderer environment index (FrameParams::environment). On first request, reads from the cache or
    // bakes (dev build), afterwards only returns the index. 0 if empty, no renderer, or failure (gradient; warns once per path).
    // When reloadChanged sees the file changed, it rebakes on the next request (old GPU textures remain)
    u32 environment(const std::string& hdriPath);
    // Reflection probe file (.mkprobe relative to assets) → renderer probe index. Like environment, uploaded on first request and re-uploaded when changed
    u32 probe(const std::string& probePath);
    // Color grading LUT (scene environment post.lut, .cube relative to assets) → renderer LUT index (PostParams::lut). Like environment, uploaded on first request,
    // and re-uploaded when changed. 0 if empty or failed (no grading)
    u32 lut(const std::string& lutPath);
    // Scene environment post values → FrameParams::post (LUT file indexed via lut())
    render::PostParams postParams(const scene::PostFields& fields);
    // Scene reflection probe objects (scene::kTypeProbe, baked ones only) → FrameParams::probes. Positions in world coordinates (both hierarchical and flattened scenes).
    // Probes over the limit (setProbeLimit, project setting probeMaxCount) are dropped
    void probes(const scene::Scene& scene, std::vector<render::ReflectionProbe>& out);
    void setProbeLimit(u32 count) { m_probeLimit = std::clamp<u32>(count, 1, render::kMaxReflectionProbes); }

    // Asset hot reload (editor edit mode): rereads model files that were ever requested and have changed (including files that newly appeared).
    // Model addresses stay; only contents change. Returns the list of reloaded paths. On failure, keeps the previous contents.
    // Old GPU meshes aren't deleted (the renderer has no mesh deletion; they only pile up during development)
    std::vector<std::string> reloadChanged();

    // Per-kind budgets (mealkit.project, game::modelBudgets). Already loaded models are rechecked too
    void setBudgets(const ModelBudget (&budgets)[kModelKindCount]);
    const ModelBudget& budget(ModelKind kind) const { return m_budgets[static_cast<u32>(kind)]; }

    // Picks the LOD by on-screen size.
    render::MeshId pickLod(const Model& model, Vec3 cameraPos, Vec3 position, Vec3 scale) const;

    // Terrain heightmap (path relative to assets): loaded on first request, nullptr if missing/unreadable (warns once).
    // Edits (editor brushes) change map in place, then terrainChanged marks the chunks to rebuild
    Terrain* terrain(const std::string& path);
    // Replaces (or creates, not yet on disk) a terrain's heightmap: new terrains and imported PNGs in the editor. dirty = not written yet
    Terrain& setTerrain(const std::string& path, Heightmap map, bool dirty);
    void terrainChanged(Terrain& terrain, const SampleRect& rect);
    // Writes edited terrains to their files (editor scene save). false if any failed (reasons in error)
    bool saveTerrains(std::string& error);
    // Terrain chunks with LOD by distance (instance at the box center, culled per chunk by the draw list). Builds stale meshes first (outside a frame)
    void addTerrain(render::DrawList& out, Terrain& terrain, Vec3 center, Vec3 size, u32 color, u32 flags, Vec3 eye);

    // Scatter layer instances (.scatter relative to assets): loaded on first request; a missing file is an empty layer (painted later).
    // Edits change items in place and set dirty; saveScatters writes them (editor scene save)
    struct ScatterEntry {
        Scatter scatter;
        bool dirty = false;
    };
    ScatterEntry& scatter(const std::string& path);
    bool saveScatters(std::string& error);
    // A scatter object's instances (fields ScatterFields): whole cells beyond drawDistance are skipped, the rest go per instance through the draw list
    void addScatter(render::DrawList& out, const scene::Entity& layer, Vec3 eye, u32 flags = 0);

    // Hash of the content of all model files the scene references (combined with scene.contentHash() for connection checks).
    u64 hashModels(const scene::Scene& scene);

    static std::string resolve(const std::string& path); // path relative to assets → real file path
    static std::string assetsRoot();                     // assets folder near the repo/executable

private:
    bool load(const std::string& path, Model& out);
    render::MeshId uploadMesh(const render::MeshData& mesh) { return m_upload(m_uploadContext, mesh); }
    std::unordered_map<std::string, std::unique_ptr<ScatterEntry>> m_scatters;
    std::unordered_map<std::string, std::unique_ptr<Terrain>> m_terrains; // nullptr = failed (retried when the file changes)
    void (*m_replace)(void*, render::MeshId, const render::MeshData&) = nullptr;

    render::MeshId (*m_upload)(void*, const render::MeshData&) = nullptr;
    u32 (*m_uploadMaterial)(void*, const render::MaterialData&) = nullptr;
    u32 (*m_uploadTexture)(void*, const CompressedTexture&) = nullptr;
    bool (*m_compressTexture)(const void*, size_t, TextureUsage, CompressedTexture&, std::string&) = nullptr;
    u32 (*m_uploadEnvironment)(void*, const std::string&, std::string&) = nullptr;
    u32 (*m_uploadProbe)(void*, const std::string&, std::string&) = nullptr;
    u32 (*m_uploadLut)(void*, const std::string&, std::string&) = nullptr;
    void* m_uploadContext = nullptr;
    struct EnvironmentEntry {
        u32 id = 0; // 0 = failed (not retried)
        unsigned long long time = 0;
    };
    std::unordered_map<std::string, EnvironmentEntry> m_environments;
    std::unordered_map<std::string, EnvironmentEntry> m_probes;
    std::unordered_map<std::string, EnvironmentEntry> m_luts;
    u32 m_probeLimit = 8;
    render::MeshId m_box = 0xFFFFFFFFu;
    std::unordered_map<std::string, std::unique_ptr<Model>> m_models;
    std::unordered_map<std::string, unsigned long long> m_fileTimes;
    ModelBudget m_budgets[kModelKindCount] = {kDefaultBudgets[0], kDefaultBudgets[1], kDefaultBudgets[2]};
    void checkBudget(Model& model) const; // file time at load (reloadChanged)
};

} // namespace mealkit::asset
