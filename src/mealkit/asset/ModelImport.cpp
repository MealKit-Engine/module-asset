// glTF import (asset/Cooked.h importModel): only in builds with MEALKIT_ASSET_IMPORT (dev builds, editor, headless). The shipping exe reads
// cooked models only, so cgltf and meshoptimizer aren't linked into it.
#include "mealkit/asset/Cooked.h"
#include "mealkit/asset/ModelCache.h"

#if MEALKIT_ASSET_IMPORT

#include "mealkit/core/Hash.h"
#include "mealkit/core/Log.h"
#include "mealkit/core/Profile.h"
#include "mealkit/platform/Paths.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <cstring>

#pragma warning(push, 0)
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#pragma warning(pop)
#include <meshoptimizer.h>

namespace mealkit::asset {
namespace {


constexpr u32 kMaxHullPoints = 1024;

// Makes cgltf's file reading work with UTF-8 paths (Korean folder names) too.
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

// Material list within the model (vertex material is an index into this list while loading, changed to the renderer index after upload)
struct Materials {
    std::vector<const cgltf_material*> list; // nullptr = primitive without material (default)
    u16 index(const cgltf_material* m) {
        for (size_t i = 0; i < list.size(); ++i)
            if (list[i] == m) return static_cast<u16>(i);
        list.push_back(m);
        return static_cast<u16>(list.size() - 1);
    }
};

// Skin reading state: joint index remap for one skin (the first) (glTF skin.joints index → Skeleton index)
struct SkinState {
    const cgltf_skin* skin = nullptr;
    std::vector<u8> remap;              // skin.joints index → Skeleton index
    std::vector<const cgltf_node*> nodes; // Skeleton index → node
    i32 jointOf(const cgltf_node* n) const { // the closest joint among the node or its ancestors (0 if none: root)
        for (; n; n = n->parent)
            for (size_t i = 0; i < nodes.size(); ++i)
                if (nodes[i] == n) return static_cast<i32>(i);
        return 0;
    }
};

u32 packWeights(const f32 w[4]) {
    f32 sum = w[0] + w[1] + w[2] + w[3];
    if (sum <= 0.0f) return 0xFFu;
    u32 q[4], total = 0;
    for (int i = 0; i < 4; ++i) total += q[i] = static_cast<u32>(std::lround(std::clamp(w[i] / sum, 0.0f, 1.0f) * 255.0f));
    // Make the sum 255 (adjusting the largest weight)
    int big = 0;
    for (int i = 1; i < 4; ++i) if (q[i] > q[big]) big = i;
    q[big] = static_cast<u32>(static_cast<i32>(q[big]) + 255 - static_cast<i32>(total));
    return q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
}

void appendPrimitive(const cgltf_primitive& prim, const f32 world[16], MeshData& out, bool& hadNormals, Materials& materials,
                     const SkinState* skin, i32 rigidJoint) {
    if (prim.type != cgltf_primitive_type_triangles) return;
    const cgltf_accessor* positions = nullptr;
    const cgltf_accessor* normals = nullptr;
    const cgltf_accessor* colors = nullptr;
    const cgltf_accessor* texcoords = nullptr;
    const cgltf_accessor* jointsAttr = nullptr;
    const cgltf_accessor* weightsAttr = nullptr;
    for (cgltf_size i = 0; i < prim.attributes_count; ++i) {
        const cgltf_attribute& a = prim.attributes[i];
        if (a.type == cgltf_attribute_type_position) positions = a.data;
        else if (a.type == cgltf_attribute_type_normal) normals = a.data;
        else if (a.type == cgltf_attribute_type_color && a.index == 0) colors = a.data;
        else if (a.type == cgltf_attribute_type_texcoord && a.index == 0) texcoords = a.data;
        else if (a.type == cgltf_attribute_type_joints && a.index == 0) jointsAttr = a.data;
        else if (a.type == cgltf_attribute_type_weights && a.index == 0) weightsAttr = a.data;
    }
    if (!positions) return;
    hadNormals = hadNormals && normals != nullptr;
    const u16 material = materials.index(prim.material); // base color is in the material table (vertex color is COLOR_0 only)

    // Column-major matrix. If the determinant is negative it's a mirror transform, so the winding order is flipped once more.
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
        // glTF is right-handed: flip z to convert to MealKit (left-handed). Texture coordinates stay as glTF (v down, same as DX12)
        out.vertices.push_back({{wp.x, wp.y, -wp.z}, {wn.x, wn.y, -wn.z}, encodeColor(c), packHalf2(t[0], t[1]), material});
        if (skin) { // skinned model: primitives without joints attach to the closest ancestor joint (rigidJoint)
            SkinWeights sw;
            sw.joints = static_cast<u32>(rigidJoint);
            if (jointsAttr && weightsAttr) {
                cgltf_uint j[4] = {};
                f32 w[4] = {};
                cgltf_accessor_read_uint(jointsAttr, i, j, 4);
                cgltf_accessor_read_float(weightsAttr, i, w, 4);
                sw.joints = 0;
                for (int k = 0; k < 4; ++k)
                    sw.joints |= static_cast<u32>(j[k] < skin->remap.size() ? skin->remap[j[k]] : 0) << (8 * k);
                sw.weights = packWeights(w);
            }
            out.skin.push_back(sw);
        }
    }
    const cgltf_size indexCount = prim.indices ? prim.indices->count : positions->count;
    const bool flip = det >= 0.0f; // flipped once by the z inversion, and again for a mirror transform
    for (cgltf_size i = 0; i + 2 < indexCount; i += 3) {
        u32 tri[3];
        for (int k = 0; k < 3; ++k)
            tri[k] = static_cast<u32>(prim.indices ? cgltf_accessor_read_index(prim.indices, i + k) : i + k);
        out.indices.push_back(base0 + tri[0]);
        out.indices.push_back(base0 + (flip ? tri[2] : tri[1]));
        out.indices.push_back(base0 + (flip ? tri[1] : tri[2]));
    }
}

void visitNode(const cgltf_node* node, MeshData& out, bool& hadNormals, Materials& materials, const SkinState* skin) {
    if (node->mesh) {
        f32 world[16];
        cgltf_node_transform_world(node, world);
        const bool ownSkin = skin && node->skin == skin->skin;
        if (ownSkin) { // glTF: the node transform of a skinned mesh isn't used (vertices are in bind space)
            std::memset(world, 0, sizeof(world));
            world[0] = world[5] = world[10] = world[15] = 1.0f;
        }
        const i32 rigid = skin ? skin->jointOf(node) : 0;
        for (cgltf_size i = 0; i < node->mesh->primitives_count; ++i)
            appendPrimitive(node->mesh->primitives[i], world, out, hadNormals, materials, skin, rigid);
    }
    for (cgltf_size i = 0; i < node->children_count; ++i) visitNode(node->children[i], out, hadNormals, materials, skin);
}

// glTF column-major matrix → row vector convention Mat4 (memory as is: column i becomes row i)
Mat4 fromGltf(const f32 m[16]) {
    Mat4 r;
    std::memcpy(&r.m[0][0], m, sizeof(f32) * 16);
    return r;
}

// First skin → Skeleton (reordered so parents come first). false if there are too many joints
bool readSkeleton(const cgltf_skin& skin, SkinState& state, Skeleton& out, std::string& why) {
    const size_t count = skin.joints_count;
    if (count == 0) return false;
    if (count > kMaxJoints) {
        why = std::to_string(count) + " joints (max " + std::to_string(kMaxJoints) + ")";
        return false;
    }
    const auto skinIndex = [&](const cgltf_node* n) -> i32 {
        for (size_t i = 0; i < count; ++i)
            if (skin.joints[i] == n) return static_cast<i32>(i);
        return -1;
    };
    std::vector<i32> parent(count, -1), depth(count, 0); // by skin index
    for (size_t i = 0; i < count; ++i)
        for (const cgltf_node* n = skin.joints[i]->parent; n; n = n->parent) {
            const i32 p = skinIndex(n);
            if (p >= 0 && parent[i] < 0) parent[i] = p;
            if (p >= 0) ++depth[i];
        }
    std::vector<size_t> order(count);
    for (size_t i = 0; i < count; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return depth[a] < depth[b]; });
    state.skin = &skin;
    state.remap.assign(count, 0);
    state.nodes.resize(count);
    for (size_t k = 0; k < count; ++k) {
        state.remap[order[k]] = static_cast<u8>(k);
        state.nodes[k] = skin.joints[order[k]];
    }
    out.joints.resize(count);
    for (size_t k = 0; k < count; ++k) {
        const size_t i = order[k];
        const cgltf_node* n = skin.joints[i];
        Skeleton::Joint& j = out.joints[k];
        j.name = n->name ? n->name : "joint" + std::to_string(k);
        j.parent = parent[i] >= 0 ? static_cast<i32>(state.remap[static_cast<size_t>(parent[i])]) : -1;
        if (n->has_translation) j.rest.translation = {n->translation[0], n->translation[1], n->translation[2]};
        if (n->has_rotation) j.rest.rotation = {n->rotation[0], n->rotation[1], n->rotation[2], n->rotation[3]};
        if (n->has_scale) j.rest.scale = {n->scale[0], n->scale[1], n->scale[2]};
        f32 ib[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        if (skin.inverse_bind_matrices) cgltf_accessor_read_float(skin.inverse_bind_matrices, i, ib, 16);
        j.inverseBind = fromGltf(ib);
    }
    const cgltf_node* top = state.nodes[0]->parent; // node above the root joint (Blender armature object)
    if (top) {
        f32 m[16];
        cgltf_node_transform_world(top, m);
        out.root = fromGltf(m);
    }
    return true;
}

// glTF animation → clip baked at 30 Hz. Channels that move non-joint nodes and morph weights are dropped.
// Interpolation: linear (rotations nlerp), step, and cubic spline as linear between key values (tangents unused)
AnimationClip bakeClip(const cgltf_animation& anim, const SkinState& state, const Skeleton& skeleton, size_t index) {
    AnimationClip clip;
    clip.name = anim.name && *anim.name ? anim.name : "clip" + std::to_string(index);
    f32 duration = 0.0f;
    for (cgltf_size c = 0; c < anim.channels_count; ++c) {
        const cgltf_accessor* input = anim.channels[c].sampler->input;
        if (input->count) {
            f32 last = 0.0f;
            cgltf_accessor_read_float(input, input->count - 1, &last, 1);
            duration = std::max(duration, last);
        }
    }
    const u32 joints = static_cast<u32>(skeleton.joints.size());
    clip.frames = static_cast<u32>(std::ceil(duration * static_cast<f32>(kClipFrameRate) - 1e-3f)) + 1;
    clip.duration = static_cast<f32>(clip.frames - 1) / static_cast<f32>(kClipFrameRate);
    clip.poses.resize(static_cast<size_t>(clip.frames) * joints);
    for (u32 f = 0; f < clip.frames; ++f)
        for (u32 j = 0; j < joints; ++j) clip.poses[static_cast<size_t>(f) * joints + j] = skeleton.joints[j].rest;
    for (cgltf_size c = 0; c < anim.channels_count; ++c) {
        const cgltf_animation_channel& ch = anim.channels[c];
        i32 joint = -1;
        for (size_t k = 0; k < state.nodes.size(); ++k)
            if (state.nodes[k] == ch.target_node) joint = static_cast<i32>(k);
        const cgltf_animation_path_type path = ch.target_path;
        if (joint < 0 || path == cgltf_animation_path_type_weights || path == cgltf_animation_path_type_invalid) continue;
        const cgltf_animation_sampler& s = *ch.sampler;
        const size_t keys = s.input->count;
        if (keys == 0) continue;
        const int comps = path == cgltf_animation_path_type_rotation ? 4 : 3;
        const bool cubic = s.interpolation == cgltf_interpolation_type_cubic_spline;
        std::vector<f32> times(keys), values(keys * comps);
        for (size_t k = 0; k < keys; ++k) {
            cgltf_accessor_read_float(s.input, k, &times[k], 1);
            cgltf_accessor_read_float(s.output, cubic ? k * 3 + 1 : k, &values[k * comps], comps);
        }
        for (u32 f = 0; f < clip.frames; ++f) {
            const f32 t = static_cast<f32>(f) / static_cast<f32>(kClipFrameRate);
            size_t k = 0;
            while (k + 1 < keys && times[k + 1] <= t) ++k;
            size_t k1 = std::min(k + 1, keys - 1);
            f32 u = 0.0f;
            if (k1 != k && t > times[k] && s.interpolation != cgltf_interpolation_type_step)
                u = std::clamp((t - times[k]) / std::max(times[k1] - times[k], 1e-6f), 0.0f, 1.0f);
            const f32* a = &values[k * comps];
            const f32* b = &values[k1 * comps];
            JointPose& pose = clip.poses[static_cast<size_t>(f) * joints + static_cast<size_t>(joint)];
            if (path == cgltf_animation_path_type_rotation)
                pose.rotation = nlerp({a[0], a[1], a[2], a[3]}, {b[0], b[1], b[2], b[3]}, u);
            else {
                const Vec3 v = lerp(Vec3{a[0], a[1], a[2]}, Vec3{b[0], b[1], b[2]}, u);
                if (path == cgltf_animation_path_type_translation) pose.translation = v;
                else pose.scale = v;
            }
        }
    }
    return clip;
}

// Material features outside the supported range (warned and ignored on load, docs/CODING_MODEL.md "Models, materials, textures")
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

// Encoded bytes of a glTF image (buffer view inside the glb, or a file next to the .gltf)
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
    return false; // data URIs aren't read (Blender embeds in the glb)
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

// Cleans up vertex format V (first field is position)
template <class V>
void optimizeVertices(std::vector<V>& in, std::vector<u32>& indices) {
    const size_t indexCount = indices.size();
    std::vector<u32> remap(in.size());
    const size_t unique = meshopt_generateVertexRemap(remap.data(), indices.data(), indexCount, in.data(), in.size(), sizeof(V));
    std::vector<V> vertices(unique);
    meshopt_remapVertexBuffer(vertices.data(), in.data(), in.size(), sizeof(V), remap.data());
    meshopt_remapIndexBuffer(indices.data(), indices.data(), indexCount, remap.data());
    meshopt_optimizeVertexCache(indices.data(), indices.data(), indexCount, unique);
    meshopt_optimizeOverdraw(indices.data(), indices.data(), indexCount, reinterpret_cast<const f32*>(vertices.data()), unique, sizeof(V), 1.05f);
    const size_t used = meshopt_optimizeVertexFetch(vertices.data(), indices.data(), indexCount, vertices.data(), unique, sizeof(V));
    vertices.resize(used);
    in = std::move(vertices);
}

struct SkinnedVertex {
    Vertex v;
    SkinWeights s;
};

void optimize(MeshData& mesh) {
    if (mesh.skin.empty()) {
        optimizeVertices(mesh.vertices, mesh.indices);
        return;
    }
    std::vector<SkinnedVertex> both(mesh.vertices.size());
    for (size_t i = 0; i < both.size(); ++i) both[i] = {mesh.vertices[i], mesh.skin[i]};
    optimizeVertices(both, mesh.indices);
    mesh.vertices.resize(both.size());
    mesh.skin.resize(both.size());
    for (size_t i = 0; i < both.size(); ++i) {
        mesh.vertices[i] = both[i].v;
        mesh.skin[i] = both[i].s;
    }
}

// LOD: a mesh with reduced triangles. Not created if it doesn't reduce enough.
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

} // namespace

bool importModel(const std::string& path, CookedModel& model, TextureCompressor compress, std::string& error) {
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
    // Skin: only the first skin (with multiple skins, remaining meshes attach to the first skin's root in bind pose)
    SkinState skinState;
    model = {};
    if (data->skins_count) {
        std::string why;
        if (readSkeleton(data->skins[0], skinState, model.skeleton, why)) {
            model.bones = static_cast<u32>(model.skeleton.joints.size());
            for (cgltf_size a = 0; a < data->animations_count; ++a)
                model.clips.push_back(bakeClip(data->animations[a], skinState, model.skeleton, a));
            if (data->skins_count > 1) MEALKIT_WARN("Model %s: %zu skins, only the first is animated", path.c_str(), data->skins_count);
        } else if (!why.empty()) {
            MEALKIT_WARN("Model %s: skin ignored: %s (drawn in the bind pose)", path.c_str(), why.c_str());
            model.bones = static_cast<u32>(data->skins[0].joints_count);
        }
    }
    const SkinState* skin = model.skeleton.valid() ? &skinState : nullptr;
    const cgltf_scene* sceneNode = data->scene ? data->scene : (data->scenes_count ? &data->scenes[0] : nullptr);
    if (sceneNode) {
        for (cgltf_size i = 0; i < sceneNode->nodes_count; ++i) visitNode(sceneNode->nodes[i], mesh, hadNormals, materials, skin);
    } else {
        for (cgltf_size i = 0; i < data->nodes_count; ++i)
            if (!data->nodes[i].parent) visitNode(&data->nodes[i], mesh, hadNormals, materials, skin);
    }

    // Materials and their textures (compressed here, uploaded by ModelCache). Vertex material stays the index into model.materials
    for (const cgltf_material* m : materials.list) if (m) warnUnsupported(path, *m);
    {
        std::vector<std::pair<const cgltf_image*, TextureUsage>> keys;
        std::vector<u32> loaded;
        const auto texture = [&](const cgltf_texture_view& view, TextureUsage usage) {
            const cgltf_image* image = view.texture ? view.texture->image : nullptr;
            if (!image || !compress) return kNoTexture;
            for (size_t i = 0; i < keys.size(); ++i)
                if (keys[i].first == image && keys[i].second == usage) return loaded[i];
            std::string storage, why;
            const u8* bytes = nullptr;
            size_t size = 0;
            u32 index = kNoTexture;
            CompressedTexture t;
            if (!imageBytes(*image, path, storage, &bytes, &size)) why = "image data not found (data URIs are not read)";
            else if (compress(bytes, size, usage, t, why)) {
                index = static_cast<u32>(model.textures.size());
                model.textures.push_back(std::move(t));
            }
            if (index == kNoTexture) MEALKIT_WARN("Model %s: texture %s: %s", path.c_str(), image->name ? image->name : "?", why.c_str());
            keys.emplace_back(image, usage);
            loaded.push_back(index);
            return index;
        };
        for (const cgltf_material* m : materials.list) {
            MaterialData md;
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
                std::vector<std::string> problems; // style preset material features (Blender custom properties → extras)
                applyMaterialExtras(m->extras.data, md, problems);
                for (const std::string& p : problems)
                    MEALKIT_WARN("Model %s, material '%s': %s", path.c_str(), m->name ? m->name : "?", p.c_str());
            }
            model.materials.push_back(md);
        }
    }
    cgltf_free(data);
    if (mesh.indices.empty()) {
        error = "no triangle meshes";
        return false;
    }
    if (!hadNormals) computeNormals(mesh);

    // Physics hull points and bounds: extracted deterministically from the original vertices before optimization (not dependent on meshoptimizer output)
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
    const bool skinned = !mesh.skin.empty();
    std::vector<MeshData>& lods = model.lods;
    lods.push_back(std::move(mesh));
    MeshData lod;
    if (!skinned && makeLod(lods[0], 0.5f, 0.02f, lod)) { // skinned meshes get no LODs (only a few characters)
        lods.push_back(std::move(lod));
        MeshData lod2;
        if (makeLod(lods[1], 0.4f, 0.05f, lod2)) lods.push_back(std::move(lod2));
    }
    return true;
}

} // namespace mealkit::asset

#endif // MEALKIT_ASSET_IMPORT
