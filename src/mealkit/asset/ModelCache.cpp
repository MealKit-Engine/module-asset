#include "mealkit/asset/ModelCache.h"

#include "mealkit/core/Hash.h"
#include "mealkit/core/Log.h"
#include "mealkit/core/Profile.h"
#include "mealkit/platform/Paths.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#pragma warning(push, 0)
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#pragma warning(pop)
#include <meshoptimizer.h>

namespace mealkit::asset {
namespace {

using render::MeshData;
using render::Vertex;

constexpr u32 kMaxHullPoints = 1024;

// cgltf의 파일 읽기를 UTF-8 경로(한글 폴더)에서도 동작하게 바꾼다.
cgltf_result readFile(const cgltf_memory_options*, const cgltf_file_options*, const char* path, cgltf_size* size,
                      void** data) {
    FILE* f = openFile(path, "rb");
    if (!f) return cgltf_result_file_not_found;
    fseek(f, 0, SEEK_END);
    const long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* buffer = std::malloc(static_cast<size_t>(length));
    const size_t read = buffer ? fread(buffer, 1, static_cast<size_t>(length), f) : 0;
    fclose(f);
    if (read != static_cast<size_t>(length)) {
        std::free(buffer);
        return cgltf_result_io_error;
    }
    *size = static_cast<cgltf_size>(length);
    *data = buffer;
    return cgltf_result_success;
}

void releaseFile(const cgltf_memory_options*, const cgltf_file_options*, void* data) { std::free(data); }

u32 encodeColor(const f32 linear[4]) {
    auto channel = [](f32 v) {
        const f32 srgb = std::pow(std::clamp(v, 0.0f, 1.0f), 1.0f / 2.2f);
        return static_cast<u32>(std::lround(srgb * 255.0f));
    };
    return channel(linear[0]) | (channel(linear[1]) << 8) | (channel(linear[2]) << 16) | (channel(linear[3]) << 24);
}

// 모델 안의 머티리얼 목록 (정점의 material은 불러오는 동안 이 목록의 번호, 렌더러에 올린 뒤 렌더러 번호로 바꾼다)
struct Materials {
    std::vector<const cgltf_material*> list; // nullptr = 머티리얼 없는 프리미티브 (기본)
    u16 index(const cgltf_material* m) {
        for (size_t i = 0; i < list.size(); ++i)
            if (list[i] == m) return static_cast<u16>(i);
        list.push_back(m);
        return static_cast<u16>(list.size() - 1);
    }
};

void appendPrimitive(const cgltf_primitive& prim, const f32 world[16], MeshData& out, bool& hadNormals, Materials& materials) {
    if (prim.type != cgltf_primitive_type_triangles) return;
    const cgltf_accessor* positions = nullptr;
    const cgltf_accessor* normals = nullptr;
    const cgltf_accessor* colors = nullptr;
    const cgltf_accessor* texcoords = nullptr;
    for (cgltf_size i = 0; i < prim.attributes_count; ++i) {
        const cgltf_attribute& a = prim.attributes[i];
        if (a.type == cgltf_attribute_type_position) positions = a.data;
        else if (a.type == cgltf_attribute_type_normal) normals = a.data;
        else if (a.type == cgltf_attribute_type_color && a.index == 0) colors = a.data;
        else if (a.type == cgltf_attribute_type_texcoord && a.index == 0) texcoords = a.data;
    }
    if (!positions) return;
    hadNormals = hadNormals && normals != nullptr;
    const u16 material = materials.index(prim.material); // 기본 색은 머티리얼 표에 (정점 색은 COLOR_0만)

    // 열 우선(column-major) 행렬. 행렬식이 음수면 거울 변환이라 감는 순서를 한 번 더 뒤집는다.
    const f32* m = world;
    const f32 det =
        m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]);

    const u32 base0 = static_cast<u32>(out.vertices.size());
    for (cgltf_size i = 0; i < positions->count; ++i) {
        f32 p[3] = {}, n[3] = {0, 1, 0}, c[4] = {1, 1, 1, 1}, t[2] = {};
        cgltf_accessor_read_float(positions, i, p, 3);
        if (normals) cgltf_accessor_read_float(normals, i, n, 3);
        if (colors) cgltf_accessor_read_float(colors, i, c, cgltf_num_components(colors->type));
        if (texcoords) cgltf_accessor_read_float(texcoords, i, t, 2);
        const Vec3 wp{m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
                      m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]};
        const Vec3 wn = normalize(Vec3{m[0] * n[0] + m[4] * n[1] + m[8] * n[2], m[1] * n[0] + m[5] * n[1] + m[9] * n[2],
                                       m[2] * n[0] + m[6] * n[1] + m[10] * n[2]});
        // glTF는 오른손 좌표계: z를 뒤집어 MealKit(왼손)로 바꾼다. 텍스처 좌표는 glTF 그대로 (v 아래로, DX12와 같다)
        out.vertices.push_back({{wp.x, wp.y, -wp.z}, {wn.x, wn.y, -wn.z}, encodeColor(c), render::packHalf2(t[0], t[1]), material});
    }
    const cgltf_size indexCount = prim.indices ? prim.indices->count : positions->count;
    const bool flip = det >= 0.0f; // z 반전으로 한 번 뒤집히고, 거울 변환이면 다시 뒤집힌다
    for (cgltf_size i = 0; i + 2 < indexCount; i += 3) {
        u32 tri[3];
        for (int k = 0; k < 3; ++k)
            tri[k] = static_cast<u32>(prim.indices ? cgltf_accessor_read_index(prim.indices, i + k) : i + k);
        out.indices.push_back(base0 + tri[0]);
        out.indices.push_back(base0 + (flip ? tri[2] : tri[1]));
        out.indices.push_back(base0 + (flip ? tri[1] : tri[2]));
    }
}

void visitNode(const cgltf_node* node, MeshData& out, bool& hadNormals, Materials& materials) {
    if (node->mesh) {
        f32 world[16];
        cgltf_node_transform_world(node, world);
        for (cgltf_size i = 0; i < node->mesh->primitives_count; ++i)
            appendPrimitive(node->mesh->primitives[i], world, out, hadNormals, materials);
    }
    for (cgltf_size i = 0; i < node->children_count; ++i) visitNode(node->children[i], out, hadNormals, materials);
}

// 지원 범위 밖 머티리얼 기능 (불러올 때 경고하고 무시한다, docs/CODING_MODEL.md "모델·머티리얼·텍스처")
void warnUnsupported(const std::string& path, const cgltf_material& m) {
    const char* name = m.name ? m.name : "?";
    const auto warn = [&](const char* what) { MEALKIT_WARN("Model %s, material '%s': %s is not supported (ignored)", path.c_str(), name, what); };
    if (m.has_pbr_specular_glossiness) warn("specular-glossiness");
    if (m.has_clearcoat) warn("clearcoat");
    if (m.has_transmission || m.has_diffuse_transmission || m.has_volume) warn("transmission");
    if (m.has_sheen) warn("sheen");
    if (m.has_iridescence) warn("iridescence");
    if (m.has_anisotropy) warn("anisotropy");
    if (m.has_specular || m.has_ior) warn("specular / IOR");
    if (m.unlit) warn("KHR_materials_unlit");
    if (m.alpha_mode == cgltf_alpha_mode_blend) warn("alpha blend (drawn opaque; use alpha clip)");
    if (m.double_sided) warn("double sided (back faces are culled)");
    if (m.occlusion_texture.texture) warn("occlusion texture");
    const cgltf_texture_view* views[] = {&m.pbr_metallic_roughness.base_color_texture,
                                         &m.pbr_metallic_roughness.metallic_roughness_texture, &m.normal_texture, &m.emissive_texture};
    for (const cgltf_texture_view* v : views)
        if (v->texture && (v->texcoord != 0 || v->has_transform)) warn("a second UV set or texture transform");
}

// glTF 이미지의 인코딩된 바이트 (glb 안의 buffer view, 또는 .gltf 옆 파일)
bool imageBytes(const cgltf_image& image, const std::string& gltfPath, std::string& out, const u8** data, size_t* size) {
    if (image.buffer_view && image.buffer_view->buffer->data) {
        *data = static_cast<const u8*>(image.buffer_view->buffer->data) + image.buffer_view->offset;
        *size = image.buffer_view->size;
        return true;
    }
    if (image.uri && std::strncmp(image.uri, "data:", 5) != 0) {
        char decoded[1024];
        std::snprintf(decoded, sizeof(decoded), "%s", image.uri);
        cgltf_decode_uri(decoded);
        if (!readTextFile(parentDirectory(gltfPath) + "/" + decoded, out)) return false;
        *data = reinterpret_cast<const u8*>(out.data());
        *size = out.size();
        return true;
    }
    return false; // data URI는 읽지 않는다 (Blender는 glb에 넣는다)
}

void computeNormals(MeshData& mesh) {
    for (Vertex& v : mesh.vertices) v.normal = {};
    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        Vertex& a = mesh.vertices[mesh.indices[i]];
        Vertex& b = mesh.vertices[mesh.indices[i + 1]];
        Vertex& c = mesh.vertices[mesh.indices[i + 2]];
        const Vec3 n = cross(b.position - a.position, c.position - a.position);
        a.normal += n;
        b.normal += n;
        c.normal += n;
    }
    for (Vertex& v : mesh.vertices) v.normal = normalize(v.normal);
}

void optimize(MeshData& mesh) {
    const size_t indexCount = mesh.indices.size();
    std::vector<u32> remap(mesh.vertices.size());
    const size_t unique = meshopt_generateVertexRemap(remap.data(), mesh.indices.data(), indexCount,
                                                      mesh.vertices.data(), mesh.vertices.size(), sizeof(Vertex));
    std::vector<Vertex> vertices(unique);
    meshopt_remapVertexBuffer(vertices.data(), mesh.vertices.data(), mesh.vertices.size(), sizeof(Vertex),
                              remap.data());
    meshopt_remapIndexBuffer(mesh.indices.data(), mesh.indices.data(), indexCount, remap.data());
    meshopt_optimizeVertexCache(mesh.indices.data(), mesh.indices.data(), indexCount, unique);
    meshopt_optimizeOverdraw(mesh.indices.data(), mesh.indices.data(), indexCount, &vertices[0].position.x, unique,
                             sizeof(Vertex), 1.05f);
    const size_t used = meshopt_optimizeVertexFetch(vertices.data(), mesh.indices.data(), indexCount, vertices.data(),
                                                    unique, sizeof(Vertex));
    vertices.resize(used);
    mesh.vertices = std::move(vertices);
}

// LOD: 삼각형을 줄인 메시. 충분히 줄지 않으면 만들지 않는다.
bool makeLod(const MeshData& source, f32 ratio, f32 maxError, MeshData& out) {
    const size_t target = static_cast<size_t>(static_cast<f32>(source.indices.size()) * ratio) / 3 * 3;
    std::vector<u32> indices(source.indices.size());
    f32 error = 0.0f;
    const size_t count =
        meshopt_simplify(indices.data(), source.indices.data(), source.indices.size(), &source.vertices[0].position.x,
                         source.vertices.size(), sizeof(Vertex), target, maxError, 0, &error);
    if (count < 3 || count > source.indices.size() * 8 / 10) return false;
    indices.resize(count);
    meshopt_optimizeVertexCache(indices.data(), indices.data(), count, source.vertices.size());
    std::vector<Vertex> vertices(source.vertices.size());
    const size_t used = meshopt_optimizeVertexFetch(vertices.data(), indices.data(), count, source.vertices.data(),
                                                    source.vertices.size(), sizeof(Vertex));
    vertices.resize(used);
    out.vertices = std::move(vertices);
    out.indices = std::move(indices);
    return true;
}

// 렌더러에 머티리얼·텍스처를 올리는 함수 (ModelCache::setRenderer, 헤드리스는 없음)
struct Uploads {
    u32 (*material)(void*, const render::MaterialData&) = nullptr;
    u32 (*texture)(void*, const void*, size_t, TextureUsage, u32*, std::string&) = nullptr;
    void* context = nullptr;
};

bool loadGltf(const std::string& path, std::vector<MeshData>& lods, Model& model, std::string& error, const Uploads& up) {
    MEALKIT_ZONE("Load glTF");
    cgltf_options options{};
    options.file.read = readFile;
    options.file.release = releaseFile;
    cgltf_data* data = nullptr;
    cgltf_result r = cgltf_parse_file(&options, path.c_str(), &data);
    if (r == cgltf_result_success) r = cgltf_load_buffers(&options, data, path.c_str());
    if (r == cgltf_result_success) r = cgltf_validate(data);
    if (r != cgltf_result_success) {
        error = "invalid or unreadable glTF (cgltf error " + std::to_string(static_cast<int>(r)) + ")";
        if (data) cgltf_free(data);
        return false;
    }

    Hasher fileHash;
    if (data->bin) fileHash.bytes(data->bin, data->bin_size);
    for (cgltf_size i = 0; i < data->buffers_count; ++i)
        if (data->buffers[i].data) fileHash.bytes(data->buffers[i].data, data->buffers[i].size);
    fileHash.bytes(data->json, data->json_size);
    model.fileHash = fileHash.value;

    MeshData mesh;
    bool hadNormals = true;
    Materials materials;
    const cgltf_scene* sceneNode = data->scene ? data->scene : (data->scenes_count ? &data->scenes[0] : nullptr);
    if (sceneNode) {
        for (cgltf_size i = 0; i < sceneNode->nodes_count; ++i) visitNode(sceneNode->nodes[i], mesh, hadNormals, materials);
    } else {
        for (cgltf_size i = 0; i < data->nodes_count; ++i)
            if (!data->nodes[i].parent) visitNode(&data->nodes[i], mesh, hadNormals, materials);
    }

    // 머티리얼 → 렌더러 머티리얼 표 (텍스처는 캐시에서). 정점의 material을 렌더러 번호로 바꾼다
    std::vector<u16> remap(materials.list.size(), 0);
    for (const cgltf_material* m : materials.list) if (m) warnUnsupported(path, *m);
    model.materials = static_cast<u32>(materials.list.size());
    if (up.material) {
        std::vector<std::pair<const cgltf_image*, TextureUsage>> keys;
        std::vector<u32> loaded;
        const auto texture = [&](const cgltf_texture_view& view, TextureUsage usage) {
            const cgltf_image* image = view.texture ? view.texture->image : nullptr;
            if (!image || !up.texture) return render::kNoTexture;
            for (size_t i = 0; i < keys.size(); ++i)
                if (keys[i].first == image && keys[i].second == usage) return loaded[i];
            std::string storage, why;
            const u8* bytes = nullptr;
            size_t size = 0;
            u32 index = render::kNoTexture, longest = 0;
            if (!imageBytes(*image, path, storage, &bytes, &size)) why = "image data not found (data URIs are not read)";
            else index = up.texture(up.context, bytes, size, usage, &longest, why);
            if (index == render::kNoTexture) MEALKIT_WARN("Model %s: texture %s: %s", path.c_str(), image->name ? image->name : "?", why.c_str());
            model.textureSize = std::max(model.textureSize, longest);
            if (index != render::kNoTexture) ++model.textures;
            keys.emplace_back(image, usage);
            loaded.push_back(index);
            return index;
        };
        for (size_t i = 0; i < materials.list.size(); ++i) {
            const cgltf_material* m = materials.list[i];
            render::MaterialData md;
            if (m) {
                const cgltf_pbr_metallic_roughness& pbr = m->pbr_metallic_roughness;
                if (m->has_pbr_metallic_roughness) {
                    std::memcpy(md.baseColor, pbr.base_color_factor, sizeof(md.baseColor));
                    md.metallic = pbr.metallic_factor;
                    md.roughness = pbr.roughness_factor;
                    md.baseColorTexture = texture(pbr.base_color_texture, TextureUsage::Color);
                    md.metallicRoughnessTexture = texture(pbr.metallic_roughness_texture, TextureUsage::Linear);
                }
                std::memcpy(md.emissive, m->emissive_factor, sizeof(md.emissive));
                if (m->has_emissive_strength)
                    for (f32& e : md.emissive) e *= m->emissive_strength.emissive_strength;
                md.emissiveTexture = texture(m->emissive_texture, TextureUsage::Color);
                md.normalTexture = texture(m->normal_texture, TextureUsage::Normal);
                md.normalScale = m->normal_texture.scale;
                if (m->alpha_mode == cgltf_alpha_mode_mask) md.alphaCutoff = std::max(m->alpha_cutoff, 1e-3f);
            }
            remap[i] = static_cast<u16>(up.material(up.context, md));
        }
    }
    for (Vertex& v : mesh.vertices) v.material = remap.empty() ? 0 : remap[v.material];
    cgltf_free(data);
    if (mesh.indices.empty()) {
        error = "no triangle meshes";
        return false;
    }
    if (!hadNormals) computeNormals(mesh);

    // 물리 껍질 점과 경계: 최적화 전 원본 정점에서 결정론적으로 뽑는다 (meshoptimizer 결과에 의존하지 않음)
    model.boundsMin = {1e30f, 1e30f, 1e30f};
    model.boundsMax = {-1e30f, -1e30f, -1e30f};
    for (const Vertex& v : mesh.vertices) {
        model.boundsMin = {std::min(model.boundsMin.x, v.position.x), std::min(model.boundsMin.y, v.position.y),
                           std::min(model.boundsMin.z, v.position.z)};
        model.boundsMax = {std::max(model.boundsMax.x, v.position.x), std::max(model.boundsMax.y, v.position.y),
                           std::max(model.boundsMax.z, v.position.z)};
    }
    const size_t stride = std::max<size_t>(1, mesh.vertices.size() / kMaxHullPoints);
    for (size_t i = 0; i < mesh.vertices.size(); i += stride) model.hullPoints.push_back(mesh.vertices[i].position);

    optimize(mesh);
    lods.push_back(std::move(mesh));
    MeshData lod;
    if (makeLod(lods[0], 0.5f, 0.02f, lod)) {
        lods.push_back(std::move(lod));
        MeshData lod2;
        if (makeLod(lods[1], 0.4f, 0.05f, lod2)) lods.push_back(std::move(lod2));
    }
    return true;
}

} // namespace

std::string ModelCache::assetsRoot() {
    const std::string found = findAsset("assets");
    if (!found.empty()) return found;
    return (projectRoot().empty() ? executableDirectory() : projectRoot()) + "/assets";
}

std::string ModelCache::resolve(const std::string& path) {
    if (path.empty()) return {};
    if (fileExists(path)) return path;
    return findAsset("assets/" + path);
}

namespace {
// 스키마의 예산 종류: 폴더, mealkit.project 키 접두사, 기본 한도 (애드온은 프로젝트 파일 값이 없으면 이것)
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
} // namespace

std::string modelSchemaJson() {
    // 불러오기(loadGltf)가 읽는 것과 같게: 삼각형, POSITION·NORMAL·COLOR_0·TEXCOORD_0, glTF PBR(기본 색·법선·금속/거칠기·발광
    // 텍스처와 계수, 알파 마스크). 노드 변환을 구워 메시 하나로 합친다
    std::string s = R"({
  "schemaVersion": 1,
  "blender": "5.2",
  "model": {
    "format": "glb",
    "folder": "models",
    "attributes": ["POSITION", "NORMAL", "COLOR_0", "TEXCOORD_0"],
    "mergesMeshes": true,
    "maxLods": )" + std::to_string(kMaxLods) + R"(,
    "material": {"baseColorFactor": true, "vertexColor": true, "textures": true,
                 "textureInputs": ["Base Color", "Normal", "Metallic", "Roughness", "Emission Color"], "alphaClip": true},
    "skinning": false,
    "animation": false
  },
  "materialFeatures": [],
  "budgets": {
    "kinds": [)" + budgetsJson() + R"(],
    "otherFolders": "props"
  }
}
)";
    // materialFeatures: 머티리얼 extras 기능 (스타일 프리셋의 외곽선·바람·림·디졸브는 "텍스처와 머티리얼" 항목에서 더한다)
    return s;
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
    const std::string file = resolve(path);
    m_fileTimes[path] = file.empty() ? 0 : fileWriteTime(file);
    std::vector<MeshData> lods;
    std::string error = "file not found under assets/";
    if (file.empty() || !loadGltf(file, lods, model, error, {m_uploadMaterial, m_uploadTexture, m_uploadContext})) {
        MEALKIT_WARN("Model %s: %s", path.c_str(), error.c_str());
        if (!out.valid) out = std::move(model); // 처음이면 실패한 모델로 남긴다 (다시 읽기면 이전 것을 지킨다)
        return false;
    }
    model.lodCount = static_cast<u32>(lods.size());
    for (u32 i = 0; i < model.lodCount; ++i) {
        model.triangles[i] = static_cast<u32>(lods[i].indices.size() / 3);
        if (m_upload) model.lods[i] = uploadMesh(lods[i]);
    }
    model.valid = true;
    model.kind = modelKind(path);
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
    std::vector<std::string> changed;
    for (auto& [path, model] : m_models) {
        const std::string file = resolve(path);
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
    const f32 size = radius / distance; // 화면에서 차지하는 대략적인 비율
    u32 lod = size > 0.08f ? 0 : size > 0.03f ? 1 : 2;
    lod = std::min(lod, model.lodCount - 1);
    return model.lods[lod];
}

u64 ModelCache::hashModels(const scene::Scene& scene) {
    Hasher h;
    for (const scene::Entity& e : scene.entities) {
        if (e.model.empty() || e.is(scene::kTypeDecoration)) continue;
        const Model* model = get(e.model);
        h.add(model ? model->fileHash : 0ull);
    }
    return h.value;
}

} // namespace mealkit::asset
