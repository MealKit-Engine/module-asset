#include "mealkit/asset/ModelCache.h"

#include "mealkit/asset/Cooked.h"
#include "mealkit/core/Hash.h"
#include "mealkit/core/Log.h"
#include "mealkit/core/Profile.h"
#include "mealkit/platform/Paths.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mealkit::asset {

using render::MeshData;
using render::Vertex;

std::string ModelCache::assetsRoot() {
    const std::string found = findAsset("assets");
    if (!found.empty()) return found;
    return (projectRoot().empty() ? executableDirectory() : projectRoot()) + "/assets";
}

std::string ModelCache::resolve(const std::string& path) {
    if (path.empty()) return {};
    if (fileExists(path)) return path;
    return findAssetFile(path);
}

namespace {
// Schema budget kinds: folder, mealkit.project key prefix, default limits (the add-on uses these if the project file has no values)
std::string budgetsJson() {
    std::string s;
    const char* keys[kModelKindCount] = {"budgetCharacters", "budgetProps", "budgetEnvironment"};
    for (u32 k = 0; k < kModelKindCount; ++k) {
        const ModelBudget& b = kDefaultBudgets[k];
        if (k) s += ",";
        s += "\n      {\"folder\": \"" + std::string(modelKindFolder(static_cast<ModelKind>(k))) + "\", \"key\": \"" + keys[k] +
             "\", \"triangles\": " + std::to_string(b.triangles) + ", \"textureSize\": " + std::to_string(b.textureSize) +
             ", \"bones\": " + std::to_string(b.bones) + ", \"materials\": " + std::to_string(b.materials) + "}";
    }
    return s + "\n    ";
}

// extras keys of style preset material features (Blender custom property "mealkit_<key>"). Main keys (non-color) turn the feature on
struct FeatureKey {
    const char* key;
    const char* label; // Blender panel (English)
    u32 feature;       // render::kMaterial*
    bool color;        // three linear color values (otherwise a single number)
    size_t offset;     // value within render::MaterialData
};
const FeatureKey kFeatureKeys[] = {
    {"outline", "Outline width (m)", render::kMaterialOutline, false, offsetof(render::MaterialData, outlineWidth)},
    {"outlineColor", "Outline color", render::kMaterialOutline, true, offsetof(render::MaterialData, outlineColor)},
    {"wind", "Wind sway (m, vertex alpha)", render::kMaterialWind, false, offsetof(render::MaterialData, windStrength)},
    {"rim", "Rim light strength", render::kMaterialRim, false, offsetof(render::MaterialData, rimStrength)},
    {"rimColor", "Rim light color", render::kMaterialRim, true, offsetof(render::MaterialData, rimColor)},
    {"dissolve", "Dissolve edge width", render::kMaterialDissolve, false, offsetof(render::MaterialData, dissolveEdge)},
    {"dissolveColor", "Dissolve edge glow", render::kMaterialDissolve, true, offsetof(render::MaterialData, dissolveColor)},
};

const char* featureName(u32 feature) {
    return feature == render::kMaterialOutline ? "outline"
           : feature == render::kMaterialWind  ? "wind"
           : feature == render::kMaterialRim   ? "rim"
                                               : "dissolve";
}

std::string jsonNumber(f32 v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(v));
    return buf;
}

// Schema materialFeatures: the add-on panel makes an "Add" button per key. feature is the feature (main key) the key belongs to, default is the engine default
std::string featuresJson() {
    const render::MaterialData defaults;
    std::string s;
    for (const FeatureKey& k : kFeatureKeys) {
        const f32* value = reinterpret_cast<const f32*>(reinterpret_cast<const u8*>(&defaults) + k.offset);
        const std::string def = k.color ? "[" + jsonNumber(value[0]) + ", " + jsonNumber(value[1]) + ", " + jsonNumber(value[2]) + "]"
                                        : jsonNumber(value[0]);
        s += std::string(&k == kFeatureKeys ? "" : ",") + "\n    {\"key\": \"" + k.key + "\", \"label\": \"" + k.label +
             "\", \"feature\": \"" + featureName(k.feature) + "\", \"color\": " + (k.color ? "true" : "false") +
             ", \"default\": " + def + "}";
    }
    return s + "\n  ";
}

// Top-level keys and values of an extras JSON object. Only numbers/number arrays are read as values (numbers); other values (strings/objects/booleans) get empty numbers.
// Stops where the format breaks (it's JSON written by the Blender exporter, so no strict validation)
struct ExtraValue {
    std::string key;
    std::vector<f32> numbers;
};

std::vector<ExtraValue> parseExtras(const char* json) {
    std::vector<ExtraValue> out;
    const char* p = json;
    const auto ws = [&] { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p; };
    const auto skipString = [&](std::string* text) { // p is the opening quote
        ++p;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1]) ++p;
            if (text) text->push_back(*p);
            ++p;
        }
        if (*p == '"') ++p;
    };
    const auto number = [&](f32& v) {
        char* end = nullptr;
        v = std::strtof(p, &end);
        if (end == p) return false;
        p = end;
        return true;
    };
    std::function<void()> skipValue = [&] {
        ws();
        if (*p == '"') return skipString(nullptr);
        if (*p == '{' || *p == '[') {
            const char close = *p == '{' ? '}' : ']';
            ++p;
            for (ws(); *p && *p != close; ws()) {
                if (*p == ',' || *p == ':') ++p;
                else skipValue();
            }
            if (*p) ++p;
            return;
        }
        while (*p && *p != ',' && *p != '}' && *p != ']') ++p; // numbers, true, false, null
    };
    ws();
    if (*p != '{') return out;
    ++p;
    for (;;) {
        ws();
        if (*p != '"') break;
        ExtraValue v;
        skipString(&v.key);
        ws();
        if (*p != ':') break;
        ++p;
        ws();
        f32 n = 0.0f;
        if (*p == '[') {
            const char* start = p++;
            bool numeric = true;
            for (ws(); *p && *p != ']'; ws()) {
                if (*p == ',') { ++p; continue; }
                if (!number(n)) { numeric = false; break; }
                v.numbers.push_back(n);
            }
            if (!numeric) {
                v.numbers.clear();
                p = start;
                skipValue();
            } else if (*p) {
                ++p;
            }
        } else if (!(*p == '-' || (*p >= '0' && *p <= '9')) || !number(n)) {
            skipValue();
        } else {
            v.numbers.push_back(n);
        }
        out.push_back(std::move(v));
        ws();
        if (*p != ',') break;
        ++p;
    }
    return out;
}
} // namespace

std::string modelSchemaJson() {
    // Same as what loading (loadGltf) reads: triangles, POSITION/NORMAL/COLOR_0/TEXCOORD_0, glTF PBR (base color/normal/metal-roughness/emissive
    // textures and factors, alpha mask). Bakes node transforms and merges into one mesh
    std::string s = R"({
  "schemaVersion": 1,
  "blender": "5.2",
  "model": {
    "format": "glb",
    "folder": "models",
    "attributes": ["POSITION", "NORMAL", "COLOR_0", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"],
    "mergesMeshes": true,
    "maxLods": )" + std::to_string(kMaxLods) + R"(,
    "material": {"baseColorFactor": true, "vertexColor": true, "textures": true,
                 "textureInputs": ["Base Color", "Normal", "Metallic", "Roughness", "Emission Color"], "alphaClip": true},
    "skinning": true,
    "maxJoints": )" + std::to_string(kMaxJoints) + R"(,
    "animation": true
  },
  "materialFeatures": [)" + featuresJson() + R"(],
  "budgets": {
    "kinds": [)" + budgetsJson() + R"(],
    "otherFolders": "props"
  }
}
)";
    return s;
}

void applyMaterialExtras(const char* json, render::MaterialData& material, std::vector<std::string>& problems) {
    if (!json) return;
    for (const ExtraValue& v : parseExtras(json)) {
        if (v.key.rfind("mealkit_", 0) != 0) continue;
        const std::string name = v.key.substr(8);
        const FeatureKey* key = nullptr;
        for (const FeatureKey& k : kFeatureKeys)
            if (name == k.key) key = &k;
        if (!key) {
            problems.push_back("unknown engine feature '" + v.key + "' (ignored)");
            continue;
        }
        const size_t want = key->color ? 3 : 1;
        if (v.numbers.size() != want) {
            problems.push_back("'" + v.key + "' needs " + (key->color ? "a color (3 numbers)" : "a number") + " (ignored)");
            continue;
        }
        f32* target = reinterpret_cast<f32*>(reinterpret_cast<u8*>(&material) + key->offset);
        for (size_t i = 0; i < want; ++i) target[i] = std::max(v.numbers[i], 0.0f);
        if (!key->color) material.features |= key->feature; // only main keys turn features on (color alone leaves it off by default)
    }
}

u32 ModelCache::environment(const std::string& hdriPath) {
    if (hdriPath.empty() || !m_uploadEnvironment) return 0;
    auto it = m_environments.find(hdriPath);
    if (it != m_environments.end()) return it->second.id;
    EnvironmentEntry entry;
    std::string file = resolve(hdriPath);
    if (file.empty()) file = resolve(hdriPath + kCookedEnvironmentSuffix); // baked by the cook step (pack files)
    std::string error = "file not found under assets/";
    entry.time = file.empty() ? 0 : fileWriteTime(file);
    if (!file.empty()) entry.id = m_uploadEnvironment(m_uploadContext, file, error);
    if (!entry.id) MEALKIT_WARN("Sky HDRI %s: %s (using the sky colors)", hdriPath.c_str(), error.c_str());
    m_environments.emplace(hdriPath, entry);
    return entry.id;
}

u32 ModelCache::probe(const std::string& probePath) {
    if (probePath.empty() || !m_uploadProbe) return 0;
    auto it = m_probes.find(probePath);
    if (it != m_probes.end()) return it->second.id;
    EnvironmentEntry entry;
    const std::string file = resolve(probePath);
    std::string error = "file not found under assets/ (bake the probe in the editor)";
    entry.time = file.empty() ? 0 : fileWriteTime(file);
    if (!file.empty()) entry.id = m_uploadProbe(m_uploadContext, file, error);
    if (!entry.id) MEALKIT_WARN("Reflection probe %s: %s (using the sky reflection)", probePath.c_str(), error.c_str());
    m_probes.emplace(probePath, entry);
    return entry.id;
}

u32 ModelCache::lut(const std::string& lutPath) {
    if (lutPath.empty() || !m_uploadLut) return 0;
    auto it = m_luts.find(lutPath);
    if (it != m_luts.end()) return it->second.id;
    EnvironmentEntry entry;
    const std::string file = resolve(lutPath);
    std::string error = "file not found under assets/";
    entry.time = file.empty() ? 0 : fileWriteTime(file);
    if (!file.empty()) entry.id = m_uploadLut(m_uploadContext, file, error);
    if (!entry.id) MEALKIT_WARN("Color LUT %s: %s (no color grading)", lutPath.c_str(), error.c_str());
    m_luts.emplace(lutPath, entry);
    return entry.id;
}

render::PostParams ModelCache::postParams(const scene::PostFields& f) {
    render::PostParams p;
    p.toneMapping = f.toneMapping;
    p.punchy = f.punchy;
    p.exposure = f.exposure;
    p.lut = lut(f.lut);
    p.bloomIntensity = f.bloomIntensity;
    p.bloomThreshold = f.bloomThreshold;
    p.bloomTint = f.bloomTint;
    p.vignette = f.vignette;
    p.vignetteSmoothness = f.vignetteSmoothness;
    p.grain = f.grain;
    p.chromaticAberration = f.chromaticAberration;
    return p;
}

void ModelCache::probes(const scene::Scene& scene, std::vector<render::ReflectionProbe>& out) {
    out.clear();
    for (const scene::Entity& e : scene.entities) {
        if (!e.is(scene::kTypeProbe) || out.size() >= m_probeLimit) continue;
        const u32 id = probe(scene::probeFile(e));
        const scene::Transform world = scene::worldTransform(scene, e);
        if (id) out.push_back({world.position, Vec3{std::fabs(world.scale.x), std::fabs(world.scale.y), std::fabs(world.scale.z)} * 0.5f, id});
    }
}

const Model* ModelCache::get(const std::string& path) {
    if (path.empty()) return nullptr;
    auto it = m_models.find(path);
    if (it != m_models.end()) return it->second->valid ? it->second.get() : nullptr;

    auto model = std::make_unique<Model>();
    load(path, *model);
    const Model* result = model->valid ? model.get() : nullptr;
    m_models.emplace(path, std::move(model));
    return result;
}

bool ModelCache::load(const std::string& path, Model& out) {
    Model model;
    model.path = path;
    // Source first (editor, dev builds), then the cooked file next to it (pack files, asset/Cooked.h)
    const std::string file = resolve(path);
    const std::string cookedFile = file.empty() ? resolve(path + kCookedModelSuffix) : std::string();
    m_fileTimes[path] = file.empty() && cookedFile.empty() ? 0 : fileWriteTime(file.empty() ? cookedFile : file);
    CookedModel cooked;
    std::string error = "file not found under assets/";
    bool ok = false;
    if (!cookedFile.empty()) {
        std::string bytes;
        ok = readTextFile(cookedFile, bytes) && readCookedModel(bytes, cooked, error);
    } else if (!file.empty()) {
#if MEALKIT_ASSET_IMPORT
        ok = importModel(file, cooked, m_compressTexture, error);
#else
        error = "source models are not read by shipping builds (cook the game again)";
#endif
    }
    if (!ok) {
        MEALKIT_WARN("Model %s: %s", path.c_str(), error.c_str());
        if (!out.valid) out = std::move(model); // first time, keep the failed model (on reload, keep the previous one)
        return false;
    }
    model.fileHash = cooked.fileHash;
    model.boundsMin = cooked.boundsMin;
    model.boundsMax = cooked.boundsMax;
    model.hullPoints = std::move(cooked.hullPoints);
    model.bones = cooked.bones;
    model.skeleton = std::move(cooked.skeleton);
    model.clips = std::move(cooked.clips);
    model.materials = static_cast<u32>(cooked.materials.size());

    // Textures → materials → vertex material indices into the renderer tables
    std::vector<u32> textureIds(cooked.textures.size(), render::kNoTexture);
    for (size_t i = 0; i < cooked.textures.size(); ++i) {
        const CompressedTexture& t = cooked.textures[i];
        model.textureSize = std::max({model.textureSize, t.width, t.height});
        if (m_uploadTexture) textureIds[i] = m_uploadTexture(m_uploadContext, t);
        if (!m_uploadTexture || textureIds[i] != render::kNoTexture) ++model.textures;
    }
    std::vector<u16> remap(cooked.materials.size(), 0);
    if (m_uploadMaterial)
        for (size_t i = 0; i < cooked.materials.size(); ++i) {
            render::MaterialData md = cooked.materials[i];
            for (u32* t : {&md.baseColorTexture, &md.normalTexture, &md.metallicRoughnessTexture, &md.emissiveTexture})
                *t = *t < textureIds.size() ? textureIds[*t] : render::kNoTexture;
            remap[i] = static_cast<u16>(m_uploadMaterial(m_uploadContext, md));
        }
    std::vector<MeshData>& lods = cooked.lods;
    for (MeshData& lod : lods)
        for (Vertex& v : lod.vertices) v.material = v.material < remap.size() ? remap[v.material] : 0;
    lods.resize(std::min<size_t>(lods.size(), kMaxLods));
    model.lodCount = static_cast<u32>(lods.size());
    for (u32 i = 0; i < model.lodCount; ++i) {
        model.triangles[i] = static_cast<u32>(lods[i].indices.size() / 3);
        if (m_upload) model.lods[i] = uploadMesh(lods[i]);
    }
    model.valid = true;
    model.kind = modelKind(path);
    if (model.skinned())
        MEALKIT_INFO("Model %s: %u triangles, skinned (%u joints, %zu clips)", path.c_str(), model.triangles[0], model.bones, model.clips.size());
    else
        MEALKIT_INFO("Model %s: %u triangles, %u LOD(s)%s", path.c_str(), model.triangles[0], model.lodCount,
                     model.lodCount > 1 ? "" : " (too simple to reduce)");
    checkBudget(model);
    out = std::move(model);
    return true;
}

std::vector<std::string> budgetProblems(const Model& model, const ModelBudget& budget) {
    std::vector<std::string> out;
    const auto over = [&](u32 used, u32 limit, const char* what) {
        if (used > limit) out.push_back(std::to_string(used) + " " + what + " (budget " + std::to_string(limit) + ")");
    };
    over(model.triangles[0], budget.triangles, "triangles");
    over(model.materials, budget.materials, "materials");
    over(model.bones, budget.bones, "bones");
    over(model.textureSize, budget.textureSize, "px texture");
    return out;
}

void ModelCache::checkBudget(Model& model) const {
    model.budgetProblems = budgetProblems(model, budget(model.kind));
    for (const std::string& p : model.budgetProblems)
        MEALKIT_WARN("Model %s is over the %s budget: %s", model.path.c_str(), modelKindFolder(model.kind), p.c_str());
}

void ModelCache::setBudgets(const ModelBudget (&budgets)[kModelKindCount]) {
    std::copy(std::begin(budgets), std::end(budgets), std::begin(m_budgets));
    for (auto& [path, model] : m_models)
        if (model->valid) checkBudget(*model);
}

std::vector<std::string> ModelCache::reloadChanged() {
    // Sky HDRI: erasing changed files makes the next environment() request rebake
    for (auto it = m_environments.begin(); it != m_environments.end();) {
        std::string file = resolve(it->first);
        if (file.empty()) file = resolve(it->first + kCookedEnvironmentSuffix);
        if ((file.empty() ? 0 : fileWriteTime(file)) != it->second.time) it = m_environments.erase(it);
        else ++it;
    }
    for (auto it = m_luts.begin(); it != m_luts.end();) { // re-exported LUTs
        const std::string file = resolve(it->first);
        if ((file.empty() ? 0 : fileWriteTime(file)) != it->second.time) it = m_luts.erase(it);
        else ++it;
    }
    for (auto it = m_probes.begin(); it != m_probes.end();) { // rebaked reflection probes
        const std::string file = resolve(it->first);
        if ((file.empty() ? 0 : fileWriteTime(file)) != it->second.time) it = m_probes.erase(it);
        else ++it;
    }
    for (auto it = m_terrains.begin(); it != m_terrains.end();) { // re-exported heightmaps (edited ones are kept)
        const std::string file = resolve(it->first);
        const unsigned long long time = file.empty() ? 0 : fileWriteTime(file);
        const bool same = it->second ? (it->second->dirty || time == it->second->fileTime) : time == 0;
        if (same) ++it;
        else it = m_terrains.erase(it); // reloaded on the next request (old GPU meshes remain, like models)
    }
    std::vector<std::string> changed;
    for (auto& [path, model] : m_models) {
        std::string file = resolve(path);
        if (file.empty()) file = resolve(path + kCookedModelSuffix);
        const unsigned long long time = file.empty() ? 0 : fileWriteTime(file);
        if (time == m_fileTimes[path]) continue;
        if (load(path, *model)) changed.push_back(path);
    }
    return changed;
}

render::MeshId ModelCache::boxMesh() {
    if (m_box == 0xFFFFFFFFu && m_upload) m_box = uploadMesh(render::makeBox());
    return m_box;
}

render::MeshId ModelCache::pickLod(const Model& model, Vec3 cameraPos, Vec3 position, Vec3 scale) const {
    const f32 maxScale = std::max({std::fabs(scale.x), std::fabs(scale.y), std::fabs(scale.z)});
    const f32 radius = length(model.boundsMax - model.boundsMin) * 0.5f * maxScale;
    const f32 distance = std::max(length(position - cameraPos), 0.001f);
    const f32 size = radius / distance; // rough fraction of the screen covered
    if (size < m_budgets[static_cast<u32>(model.kind)].cullSize) return render::kNoMesh;
    u32 lod = size > 0.08f ? 0 : size > 0.03f ? 1 : 2;
    lod = std::min(lod, model.lodCount - 1);
    return model.lods[lod];
}

void addModel(render::DrawList& out, const Model& model, render::MeshId mesh, const render::InstanceData& instance,
              const std::string& anim, f64 seconds) {
    if (mesh == render::kNoMesh) return; // pickLod: below the cull size
    if (!model.skinned()) {
        out.add(mesh, instance);
        return;
    }
    thread_local std::vector<JointPose> pose;
    thread_local std::vector<BoneMatrix> palette;
    const size_t space = anim.find(' ');
    const AnimationClip* clip = anim.empty() ? nullptr : findClip(model.clips, anim.substr(0, space));
    if (clip) {
        const f64 offset = space == std::string::npos ? 0.0 : std::atof(anim.c_str() + space + 1);
        samplePose(model.skeleton, *clip, static_cast<f32>(std::fmod(seconds + offset, 4096.0)), true, pose);
    } else {
        restPose(model.skeleton, pose);
    }
    skinPalette(model.skeleton, pose, palette);
    out.addSkinned(model.lods[0], instance, palette.data(), static_cast<u32>(palette.size()));
}

Terrain* ModelCache::terrain(const std::string& path) {
    if (path.empty()) return nullptr;
    auto it = m_terrains.find(path);
    if (it != m_terrains.end()) return it->second.get();
    std::unique_ptr<Terrain> t = std::make_unique<Terrain>();
    const std::string file = resolve(path);
    std::string error;
    if (file.empty() || !loadHeightmap(file, t->map, error)) {
        MEALKIT_WARN("Terrain %s: %s", path.c_str(), file.empty() ? "file not found" : error.c_str());
        m_terrains[path] = nullptr;
        return nullptr;
    }
    t->path = path;
    t->fileTime = fileWriteTime(file);
    Terrain* out = t.get();
    m_terrains[path] = std::move(t);
    return out;
}

Terrain& ModelCache::setTerrain(const std::string& path, Heightmap map, bool dirty) {
    std::unique_ptr<Terrain>& slot = m_terrains[path];
    if (!slot) slot = std::make_unique<Terrain>();
    slot->path = path;
    slot->map = std::move(map);
    slot->dirty = dirty;
    slot->chunks = 0; // rebuild everything (the size may differ; existing mesh ids are reused)
    return *slot;
}

void ModelCache::terrainChanged(Terrain& t, const SampleRect& rect) {
    if (rect.empty) return;
    t.dirty = true;
    if (t.chunks == 0) return;
    // A sample is shared by the chunks on both sides of it and changes its neighbours' normals
    auto chunkOf = [&](u32 s) { return std::min(s / kTerrainChunkCells, t.chunks - 1); };
    const u32 x0 = chunkOf(rect.x0 > 0 ? rect.x0 - 1 : 0), x1 = chunkOf(rect.x1 + 1);
    const u32 z0 = chunkOf(rect.z0 > 0 ? rect.z0 - 1 : 0), z1 = chunkOf(rect.z1 + 1);
    for (u32 z = z0; z <= z1; ++z)
        for (u32 x = x0; x <= x1; ++x) t.stale[z * t.chunks + x] = 1;
}

bool ModelCache::saveTerrains(std::string& error) {
    bool ok = true;
    for (auto& [path, t] : m_terrains) {
        if (!t || !t->dirty) continue;
        std::string file = resolve(path);
        if (file.empty()) { // new file next to the other assets
            file = assetsRoot() + "/" + path;
            createDirectories(file.substr(0, file.find_last_of("/\\")));
        }
        std::string reason;
        if (!saveHeightmap(file, t->map, reason)) {
            error += path + ": " + reason + "\n";
            ok = false;
            continue;
        }
        t->dirty = false;
        t->fileTime = fileWriteTime(file);
    }
    return ok;
}

void ModelCache::addTerrain(render::DrawList& out, Terrain& t, Vec3 center, Vec3 size, u32 color, u32 flags, Vec3 eye) {
    if (!m_upload || !t.map.valid()) return;
    const u32 chunks = terrainChunks(t.map);
    if (t.chunks != chunks || t.meshSize.x != size.x || t.meshSize.y != size.y || t.meshSize.z != size.z) {
        t.chunks = chunks; // first draw or new size: rebuild all
        t.meshSize = size;
        t.meshes.resize(static_cast<size_t>(chunks) * chunks * kTerrainLods, kNoMesh);
        t.stale.assign(static_cast<size_t>(chunks) * chunks, 1);
    }
    thread_local render::MeshData mesh;
    const f32 cellX = size.x / static_cast<f32>(t.map.size - 1), cellZ = size.z / static_cast<f32>(t.map.size - 1);
    const f32 chunkWidth = std::max(cellX, cellZ) * static_cast<f32>(kTerrainChunkCells);
    for (u32 cz = 0; cz < chunks; ++cz)
        for (u32 cx = 0; cx < chunks; ++cx) {
            const u32 chunk = cz * chunks + cx;
            if (t.stale[chunk]) {
                for (u32 lod = 0; lod < kTerrainLods; ++lod) {
                    buildTerrainChunk(t.map, size, cx, cz, lod, mesh);
                    render::MeshId& id = t.meshes[chunk * kTerrainLods + lod];
                    if (id == kNoMesh) id = uploadMesh(mesh);
                    else m_replace(m_uploadContext, id, mesh);
                }
                t.stale[chunk] = 0;
            }
            const Vec3 middle = center - size * 0.5f + Vec3{(static_cast<f32>(cx) + 0.5f) * cellX * kTerrainChunkCells, size.y * 0.5f,
                                                            (static_cast<f32>(cz) + 0.5f) * cellZ * kTerrainChunkCells};
            const u32 lod = terrainLod(length(middle - eye), chunkWidth);
            render::InstanceData inst;
            inst.position = center;
            inst.color = color;
            inst.flags = flags;
            out.add(t.meshes[chunk * kTerrainLods + lod], inst);
        }
}

ModelCache::ScatterEntry& ModelCache::scatter(const std::string& path) {
    std::unique_ptr<ScatterEntry>& slot = m_scatters[path];
    if (slot) return *slot;
    slot = std::make_unique<ScatterEntry>();
    const std::string file = path.empty() ? std::string() : resolve(path);
    std::string error;
    if (!file.empty() && !loadScatter(file, slot->scatter, error)) MEALKIT_WARN("Scatter %s: %s", path.c_str(), error.c_str());
    return *slot;
}

bool ModelCache::saveScatters(std::string& error) {
    bool ok = true;
    for (auto& [path, entry] : m_scatters) {
        if (!entry || !entry->dirty || path.empty()) continue;
        std::string file = resolve(path);
        if (file.empty()) {
            file = assetsRoot() + "/" + path;
            createDirectories(file.substr(0, file.find_last_of("/\\")));
        }
        std::string reason;
        if (!saveScatter(file, entry->scatter, reason)) {
            error += path + ": " + reason + "\n";
            ok = false;
            continue;
        }
        entry->dirty = false;
    }
    return ok;
}

void ModelCache::addScatter(render::DrawList& out, const scene::Entity& layer, Vec3 eye, u32 flags) {
    const scene::ScatterFields f = scene::readObject<scene::ScatterFields>(layer);
    if (f.file.empty()) return;
    Scatter& s = scatter(f.file).scatter;
    if (s.items.empty()) return;
    if (!s.grouped) groupScatter(s);
    const Model* model = f.model.empty() ? nullptr : get(f.model);
    const render::MeshId box = model ? 0u : boxMesh();
    const f32 reach = f.drawDistance + kScatterCellSize * 0.75f; // cell half diagonal
    for (const ScatterCell& cell : s.cells) {
        const Vec3 d = cell.center - eye;
        if (d.x * d.x + d.y * d.y + d.z * d.z > reach * reach) continue;
        for (u32 k = 0; k < cell.count; ++k) {
            const ScatterInstance& item = s.items[s.order[cell.first + k]];
            render::InstanceData inst;
            inst.rotation = quatFromAxisAngle({0.0f, 1.0f, 0.0f}, item.yaw);
            inst.color = model ? 0xFFFFFFFFu : layer.color;
            inst.flags = flags;
            if (model) {
                inst.position = item.position;
                inst.scale = {item.scale, item.scale, item.scale};
                out.add(pickLod(*model, eye, inst.position, inst.scale), inst);
            } else { // placeholder: a small box standing on the ground
                inst.scale = Vec3{0.3f, 0.6f, 0.3f} * item.scale;
                inst.position = item.position + Vec3{0.0f, inst.scale.y * 0.5f, 0.0f};
                out.add(box, inst);
            }
        }
    }
}

u64 ModelCache::hashModels(const scene::Scene& scene) {
    Hasher h;
    for (const scene::Entity& e : scene.entities) {
        if (e.is(scene::kTypeTerrain)) { // heightmap samples are collision data
            const Terrain* t = terrain(scene::terrainFile(e));
            h.add(t ? t->map.hash() : 0ull);
            continue;
        }
        if (e.model.empty() || e.is(scene::kTypeDecoration) || e.is(scene::kTypeProbe)) continue;
        // Only what the simulation reads (physics hull points): same on every platform, and retexturing doesn't split matches
        const Model* model = get(e.model);
        h.add(model ? static_cast<u64>(model->hullPoints.size()) : ~0ull);
        if (model) h.bytes(model->hullPoints.data(), model->hullPoints.size() * sizeof(Vec3));
    }
    return h.value;
}

} // namespace mealkit::asset
