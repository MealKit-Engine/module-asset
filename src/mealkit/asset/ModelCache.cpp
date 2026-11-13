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
#include <functional>

namespace mealkit::asset {

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
    u32 feature;       // kMaterial*
    bool color;        // three linear color values (otherwise a single number)
    size_t offset;     // value within MaterialData
};
const FeatureKey kFeatureKeys[] = {
    {"outline", "Outline width (m)", kMaterialOutline, false, offsetof(MaterialData, outlineWidth)},
    {"outlineColor", "Outline color", kMaterialOutline, true, offsetof(MaterialData, outlineColor)},
    {"wind", "Wind sway (m, vertex alpha)", kMaterialWind, false, offsetof(MaterialData, windStrength)},
    {"rim", "Rim light strength", kMaterialRim, false, offsetof(MaterialData, rimStrength)},
    {"rimColor", "Rim light color", kMaterialRim, true, offsetof(MaterialData, rimColor)},
    {"dissolve", "Dissolve edge width", kMaterialDissolve, false, offsetof(MaterialData, dissolveEdge)},
    {"dissolveColor", "Dissolve edge glow", kMaterialDissolve, true, offsetof(MaterialData, dissolveColor)},
};

const char* featureName(u32 feature) {
    return feature == kMaterialOutline ? "outline"
           : feature == kMaterialWind  ? "wind"
           : feature == kMaterialRim   ? "rim"
                                               : "dissolve";
}

std::string jsonNumber(f32 v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(v));
    return buf;
}

// Schema materialFeatures: the add-on panel makes an "Add" button per key. feature is the feature (main key) the key belongs to, default is the engine default
std::string featuresJson() {
    const MaterialData defaults;
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

void applyMaterialExtras(const char* json, MaterialData& material, std::vector<std::string>& problems) {
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
    if (hdriPath.empty() || !m_gpu) return 0;
    auto it = m_environments.find(hdriPath);
    if (it != m_environments.end()) return it->second.id;
    EnvironmentEntry entry;
    std::string file = resolve(hdriPath);
    if (file.empty()) file = resolve(hdriPath + kCookedEnvironmentSuffix); // baked by the cook step (pack files)
    std::string error = "file not found under assets/";
    entry.time = file.empty() ? 0 : fileWriteTime(file);
    if (!file.empty()) entry.id = m_gpu->createEnvironment(file, error);
    if (!entry.id) MEALKIT_WARN("Sky HDRI %s: %s (using the sky colors)", hdriPath.c_str(), error.c_str());
    m_environments.emplace(hdriPath, entry);
    return entry.id;
}

u32 ModelCache::probe(const std::string& probePath) {
    if (probePath.empty() || !m_gpu) return 0;
    auto it = m_probes.find(probePath);
    if (it != m_probes.end()) return it->second.id;
    EnvironmentEntry entry;
    const std::string file = resolve(probePath);
    std::string error = "file not found under assets/ (bake the probe in the editor)";
    entry.time = file.empty() ? 0 : fileWriteTime(file);
    if (!file.empty()) entry.id = m_gpu->createProbe(file, error);
    if (!entry.id) MEALKIT_WARN("Reflection probe %s: %s (using the sky reflection)", probePath.c_str(), error.c_str());
    m_probes.emplace(probePath, entry);
    return entry.id;
}

u32 ModelCache::lut(const std::string& lutPath) {
    if (lutPath.empty() || !m_gpu) return 0;
    auto it = m_luts.find(lutPath);
    if (it != m_luts.end()) return it->second.id;
    EnvironmentEntry entry;
    const std::string file = resolve(lutPath);
    std::string error = "file not found under assets/";
    entry.time = file.empty() ? 0 : fileWriteTime(file);
    if (!file.empty()) entry.id = m_gpu->createLut(file, error);
    if (!entry.id) MEALKIT_WARN("Color LUT %s: %s (no color grading)", lutPath.c_str(), error.c_str());
    m_luts.emplace(lutPath, entry);
    return entry.id;
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
        ok = importModel(file, cooked, m_gpu ? m_gpu->textureCompressor() : nullptr, error);
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
    std::vector<u32> textureIds(cooked.textures.size(), kNoTexture);
    for (size_t i = 0; i < cooked.textures.size(); ++i) {
        const CompressedTexture& t = cooked.textures[i];
        model.textureSize = std::max({model.textureSize, t.width, t.height});
        if (m_gpu) textureIds[i] = m_gpu->createTexture(t);
        if (!m_gpu || textureIds[i] != kNoTexture) ++model.textures;
    }
    std::vector<u16> remap(cooked.materials.size(), 0);
    if (m_gpu)
        for (size_t i = 0; i < cooked.materials.size(); ++i) {
            MaterialData md = cooked.materials[i];
            for (u32* t : {&md.baseColorTexture, &md.normalTexture, &md.metallicRoughnessTexture, &md.emissiveTexture})
                *t = *t < textureIds.size() ? textureIds[*t] : kNoTexture;
            remap[i] = static_cast<u16>(m_gpu->createMaterial(md));
        }
    std::vector<MeshData>& lods = cooked.lods;
    for (MeshData& lod : lods)
        for (Vertex& v : lod.vertices) v.material = v.material < remap.size() ? remap[v.material] : 0;
    lods.resize(std::min<size_t>(lods.size(), kMaxLods));
    model.lodCount = static_cast<u32>(lods.size());
    for (u32 i = 0; i < model.lodCount; ++i) {
        model.triangles[i] = static_cast<u32>(lods[i].indices.size() / 3);
        if (m_gpu) model.lods[i] = m_gpu->createMesh(lods[i]);
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
        const bool same = it->second ? time == it->second->fileTime : time == 0;
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

MeshId ModelCache::boxMesh() {
    if (m_box == kNoMesh && m_gpu) m_box = m_gpu->createMesh(makeBox());
    return m_box;
}

MeshId ModelCache::pickLod(const Model& model, Vec3 cameraPos, Vec3 position, Vec3 scale) const {
    const f32 maxScale = std::max({std::fabs(scale.x), std::fabs(scale.y), std::fabs(scale.z)});
    const f32 radius = length(model.boundsMax - model.boundsMin) * 0.5f * maxScale;
    const f32 distance = std::max(length(position - cameraPos), 0.001f);
    const f32 size = radius / distance; // rough fraction of the screen covered
    if (size < m_budgets[static_cast<u32>(model.kind)].cullSize) return kNoMesh;
    u32 lod = size > 0.08f ? 0 : size > 0.03f ? 1 : 2;
    lod = std::min(lod, model.lodCount - 1);
    return model.lods[lod];
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

bool ModelCache::buildTerrainMeshes(Terrain& t, Vec3 size) {
    if (!m_gpu || !t.map.valid()) return false;
    const u32 chunks = terrainChunks(t.map);
    if (t.chunks != chunks || t.meshSize.x != size.x || t.meshSize.y != size.y || t.meshSize.z != size.z) {
        t.chunks = chunks; // first draw or new size: rebuild all
        t.meshSize = size;
        t.meshes.resize(static_cast<size_t>(chunks) * chunks * kTerrainLods, kNoMesh);
        t.stale.assign(static_cast<size_t>(chunks) * chunks, 1);
    }
    thread_local MeshData mesh;
    for (u32 cz = 0; cz < chunks; ++cz)
        for (u32 cx = 0; cx < chunks; ++cx) {
            const u32 chunk = cz * chunks + cx;
            if (!t.stale[chunk]) continue;
            for (u32 lod = 0; lod < kTerrainLods; ++lod) {
                buildTerrainChunk(t.map, size, cx, cz, lod, mesh);
                MeshId& id = t.meshes[chunk * kTerrainLods + lod];
                if (id == kNoMesh) id = m_gpu->createMesh(mesh);
                else m_gpu->replaceMesh(id, mesh);
            }
            t.stale[chunk] = 0;
        }
    return true;
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
