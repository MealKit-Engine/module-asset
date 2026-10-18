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

void appendPrimitive(const cgltf_primitive& prim, const f32 world[16], MeshData& out, bool& hadNormals) {
    if (prim.type != cgltf_primitive_type_triangles) return;
    const cgltf_accessor* positions = nullptr;
    const cgltf_accessor* normals = nullptr;
    const cgltf_accessor* colors = nullptr;
    for (cgltf_size i = 0; i < prim.attributes_count; ++i) {
        const cgltf_attribute& a = prim.attributes[i];
        if (a.type == cgltf_attribute_type_position) positions = a.data;
        else if (a.type == cgltf_attribute_type_normal) normals = a.data;
        else if (a.type == cgltf_attribute_type_color && a.index == 0) colors = a.data;
    }
    if (!positions) return;
    hadNormals = hadNormals && normals != nullptr;

    f32 base[4] = {1, 1, 1, 1};
    if (prim.material && prim.material->has_pbr_metallic_roughness)
        std::memcpy(base, prim.material->pbr_metallic_roughness.base_color_factor, sizeof(base));

    // 열 우선(column-major) 행렬. 행렬식이 음수면 거울 변환이라 감는 순서를 한 번 더 뒤집는다.
    const f32* m = world;
    const f32 det =
        m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]);

    const u32 base0 = static_cast<u32>(out.vertices.size());
    for (cgltf_size i = 0; i < positions->count; ++i) {
        f32 p[3] = {}, n[3] = {0, 1, 0}, c[4] = {1, 1, 1, 1};
        cgltf_accessor_read_float(positions, i, p, 3);
        if (normals) cgltf_accessor_read_float(normals, i, n, 3);
        if (colors) cgltf_accessor_read_float(colors, i, c, cgltf_num_components(colors->type));
        const Vec3 wp{m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
                      m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]};
        const Vec3 wn = normalize(Vec3{m[0] * n[0] + m[4] * n[1] + m[8] * n[2], m[1] * n[0] + m[5] * n[1] + m[9] * n[2],
                                       m[2] * n[0] + m[6] * n[1] + m[10] * n[2]});
        const f32 color[4] = {base[0] * c[0], base[1] * c[1], base[2] * c[2], base[3] * c[3]};
        // glTF는 오른손 좌표계: z를 뒤집어 MealKit(왼손)로 바꾼다
        out.vertices.push_back({{wp.x, wp.y, -wp.z}, {wn.x, wn.y, -wn.z}, encodeColor(color)});
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

void visitNode(const cgltf_node* node, MeshData& out, bool& hadNormals) {
    if (node->mesh) {
        f32 world[16];
        cgltf_node_transform_world(node, world);
        for (cgltf_size i = 0; i < node->mesh->primitives_count; ++i)
            appendPrimitive(node->mesh->primitives[i], world, out, hadNormals);
    }
    for (cgltf_size i = 0; i < node->children_count; ++i) visitNode(node->children[i], out, hadNormals);
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

bool loadGltf(const std::string& path, std::vector<MeshData>& lods, Model& model, std::string& error) {
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
    const cgltf_scene* sceneNode = data->scene ? data->scene : (data->scenes_count ? &data->scenes[0] : nullptr);
    if (sceneNode) {
        for (cgltf_size i = 0; i < sceneNode->nodes_count; ++i) visitNode(sceneNode->nodes[i], mesh, hadNormals);
    } else {
        for (cgltf_size i = 0; i < data->nodes_count; ++i)
            if (!data->nodes[i].parent) visitNode(&data->nodes[i], mesh, hadNormals);
    }
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
    return (projectRoot().empty() ? executableDirectory() : projectRoot()) + "\\assets";
}

std::string ModelCache::resolve(const std::string& path) {
    if (path.empty()) return {};
    if (fileExists(path)) return path;
    return findAsset("assets/" + path);
}

const Model* ModelCache::get(const std::string& path) {
    if (path.empty()) return nullptr;
    auto it = m_models.find(path);
    if (it != m_models.end()) return it->second->valid ? it->second.get() : nullptr;

    auto model = std::make_unique<Model>();
    model->path = path;
    const std::string file = resolve(path);
    std::vector<MeshData> lods;
    std::string error = "file not found under assets/";
    if (!file.empty() && loadGltf(file, lods, *model, error)) {
        model->lodCount = static_cast<u32>(lods.size());
        for (u32 i = 0; i < model->lodCount; ++i) {
            model->triangles[i] = static_cast<u32>(lods[i].indices.size() / 3);
            if (m_renderer) model->lods[i] = m_renderer->createMesh(lods[i]);
        }
        model->valid = true;
        MEALKIT_INFO("Model %s: %u triangles, %u LOD(s)%s", path.c_str(), model->triangles[0], model->lodCount,
                     model->lodCount > 1 ? "" : " (too simple to reduce)");
    } else {
        MEALKIT_WARN("Model %s: %s", path.c_str(), error.c_str());
    }
    const Model* result = model->valid ? model.get() : nullptr;
    m_models.emplace(path, std::move(model));
    return result;
}

render::MeshId ModelCache::boxMesh() {
    if (m_box == 0xFFFFFFFFu && m_renderer) m_box = m_renderer->createMesh(render::makeBox());
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
