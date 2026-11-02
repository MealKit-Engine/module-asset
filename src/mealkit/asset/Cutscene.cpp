#include "mealkit/asset/Cutscene.h"

#include "mealkit/asset/Animation.h"
#include "mealkit/asset/Cooked.h"
#include "mealkit/core/Log.h"
#include "mealkit/data/Tables.h"
#include "mealkit/platform/Paths.h"

#if MEALKIT_ASSET_IMPORT
#include <cgltf.h>
#endif

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <vector>

namespace mealkit::asset {
namespace {

constexpr f64 kTableCheckSeconds = 0.25;
constexpr f32 kRadToDeg = 57.2957795f; // unit conversion

void warn(std::string* warnings, const std::string& text) {
    if (!warnings) return;
    if (!warnings->empty()) *warnings += "; ";
    *warnings += text;
}

#if MEALKIT_ASSET_IMPORT
// Rotation part of a column-major glTF matrix (scale removed) → quaternion
Quat quatFromColumns(const f32 m[16]) {
    Vec3 c0 = normalize(Vec3{m[0], m[1], m[2]}), c1 = normalize(Vec3{m[4], m[5], m[6]}), c2 = normalize(Vec3{m[8], m[9], m[10]});
    const f32 trace = c0.x + c1.y + c2.z;
    Quat q;
    if (trace > 0.0f) {
        const f32 s = std::sqrt(trace + 1.0f) * 2.0f;
        q = {(c1.z - c2.y) / s, (c2.x - c0.z) / s, (c0.y - c1.x) / s, 0.25f * s};
    } else if (c0.x > c1.y && c0.x > c2.z) {
        const f32 s = std::sqrt(1.0f + c0.x - c1.y - c2.z) * 2.0f;
        q = {0.25f * s, (c1.x + c0.y) / s, (c2.x + c0.z) / s, (c1.z - c2.y) / s};
    } else if (c1.y > c2.z) {
        const f32 s = std::sqrt(1.0f + c1.y - c0.x - c2.z) * 2.0f;
        q = {(c1.x + c0.y) / s, 0.25f * s, (c2.y + c1.z) / s, (c2.x - c0.z) / s};
    } else {
        const f32 s = std::sqrt(1.0f + c2.z - c0.x - c1.y) * 2.0f;
        q = {(c2.x + c0.z) / s, (c2.y + c1.z) / s, 0.25f * s, (c0.y - c1.x) / s};
    }
    return normalize(q);
}

// World transform of the camera node at its current TRS → MealKit coordinates (z flipped)
CameraKey keyOf(const cgltf_node& node, f32 fovDegrees) {
    f32 m[16];
    cgltf_node_transform_world(&node, m);
    const Quat q = quatFromColumns(m);
    CameraKey k;
    k.position = {m[12], m[13], -m[14]};
    k.rotation = {-q.x, -q.y, q.z, q.w}; // glTF cameras look along -z; after the flip that is +z
    k.fovDegrees = fovDegrees;
    return k;
}

// Values of one channel at time t (linear, step, cubic spline as linear between key values)
void sampleChannel(const cgltf_animation_sampler& s, int comps, f32 t, f32* out) {
    const size_t keys = s.input->count;
    const bool cubic = s.interpolation == cgltf_interpolation_type_cubic_spline;
    size_t k = 0;
    f32 time = 0.0f, next = 0.0f;
    for (; k + 1 < keys; ++k) {
        cgltf_accessor_read_float(s.input, k + 1, &next, 1);
        if (next > t) break;
    }
    const size_t k1 = std::min(k + 1, keys - 1);
    cgltf_accessor_read_float(s.input, k, &time, 1);
    cgltf_accessor_read_float(s.input, k1, &next, 1);
    f32 a[4] = {}, b[4] = {};
    cgltf_accessor_read_float(s.output, cubic ? k * 3 + 1 : k, a, static_cast<cgltf_size>(comps));
    cgltf_accessor_read_float(s.output, cubic ? k1 * 3 + 1 : k1, b, static_cast<cgltf_size>(comps));
    f32 u = 0.0f;
    if (k1 != k && t > time && s.interpolation != cgltf_interpolation_type_step)
        u = std::clamp((t - time) / std::max(next - time, 1e-6f), 0.0f, 1.0f);
    if (comps == 4) {
        const Quat q = nlerp({a[0], a[1], a[2], a[3]}, {b[0], b[1], b[2], b[3]}, u);
        out[0] = q.x, out[1] = q.y, out[2] = q.z, out[3] = q.w;
    } else
        for (int i = 0; i < comps; ++i) out[i] = a[i] + (b[i] - a[i]) * u;
}

bool readBytes(const std::string& path, std::vector<char>& out) {
    FILE* f = openFile(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(length > 0 ? static_cast<size_t>(length) : 0);
    const size_t read = out.empty() ? 0 : fread(out.data(), 1, out.size(), f);
    fclose(f);
    return read == out.size();
}
#endif // MEALKIT_ASSET_IMPORT

} // namespace

bool loadCameraClips(const std::string& file, std::vector<CameraClip>& out, std::string& error) {
    out.clear();
    const std::string_view cooked = kCookedCameraSuffix; // cooked by the cook step (pack files)
    if (file.size() > cooked.size() && file.compare(file.size() - cooked.size(), cooked.size(), cooked) == 0) {
        std::string bytes;
        if (!readTextFile(file, bytes)) return error = "can't read the file", false;
        return readCameraClips(bytes, out, error);
    }
#if !MEALKIT_ASSET_IMPORT
    return error = "source glTF files are not read by shipping builds (cook the game again)", false;
#else
    std::vector<char> bytes;
    if (!readBytes(file, bytes)) {
        error = "can't read the file";
        return false;
    }
    cgltf_options options{};
    cgltf_data* data = nullptr;
    cgltf_result r = cgltf_parse(&options, bytes.data(), bytes.size(), &data);
    if (r == cgltf_result_success) r = cgltf_load_buffers(&options, data, file.c_str()); // .glb: the embedded buffer
    if (r != cgltf_result_success) {
        error = "invalid glTF (cgltf error " + std::to_string(static_cast<int>(r)) + ")";
        if (data) cgltf_free(data);
        return false;
    }
    cgltf_node* node = nullptr;
    for (cgltf_size i = 0; i < data->nodes_count && !node; ++i)
        if (data->nodes[i].camera) node = &data->nodes[i];
    if (!node) {
        error = "no camera node";
        cgltf_free(data);
        return false;
    }
    const cgltf_camera& camera = *node->camera;
    const f32 fov = camera.type == cgltf_camera_type_perspective ? camera.data.perspective.yfov * kRadToDeg : 60.0f;
    const f32 rest[10] = {node->translation[0], node->translation[1], node->translation[2], node->rotation[0], node->rotation[1],
                          node->rotation[2],    node->rotation[3],    node->scale[0],       node->scale[1],    node->scale[2]};
    for (cgltf_size a = 0; a < data->animations_count; ++a) {
        const cgltf_animation& anim = data->animations[a];
        CameraClip clip;
        clip.name = anim.name && *anim.name ? anim.name : "clip" + std::to_string(a);
        f32 duration = 0.0f;
        bool moves = false;
        for (cgltf_size c = 0; c < anim.channels_count; ++c) {
            if (anim.channels[c].target_node != node) continue;
            moves = true;
            const cgltf_accessor* input = anim.channels[c].sampler->input;
            f32 last = 0.0f;
            if (input->count) cgltf_accessor_read_float(input, input->count - 1, &last, 1);
            duration = std::max(duration, last);
        }
        if (!moves) continue;
        const u32 frames = static_cast<u32>(std::ceil(duration * static_cast<f32>(kClipFrameRate) - 1e-3f)) + 1;
        clip.duration = static_cast<f32>(frames - 1) / static_cast<f32>(kClipFrameRate);
        for (u32 f = 0; f < frames; ++f) {
            const f32 t = static_cast<f32>(f) / static_cast<f32>(kClipFrameRate);
            for (cgltf_size c = 0; c < anim.channels_count; ++c) {
                const cgltf_animation_channel& ch = anim.channels[c];
                if (ch.target_node != node || !ch.sampler->input->count) continue;
                if (ch.target_path == cgltf_animation_path_type_translation) sampleChannel(*ch.sampler, 3, t, node->translation);
                if (ch.target_path == cgltf_animation_path_type_rotation) sampleChannel(*ch.sampler, 4, t, node->rotation);
                if (ch.target_path == cgltf_animation_path_type_scale) sampleChannel(*ch.sampler, 3, t, node->scale);
            }
            node->has_translation = node->has_rotation = node->has_scale = 1;
            clip.keys.push_back(keyOf(*node, fov));
        }
        std::copy(rest, rest + 3, node->translation);
        std::copy(rest + 3, rest + 7, node->rotation);
        std::copy(rest + 7, rest + 10, node->scale);
        out.push_back(std::move(clip));
    }
    if (out.empty()) { // still camera
        CameraClip clip;
        clip.name = "still";
        clip.keys.push_back(keyOf(*node, fov));
        out.push_back(std::move(clip));
    }
    cgltf_free(data);
    return true;
#endif
}

CameraKey sampleCamera(const CameraClip& clip, f32 seconds) {
    if (clip.keys.empty()) return {};
    const f32 frame = std::clamp(seconds * static_cast<f32>(kClipFrameRate), 0.0f, static_cast<f32>(clip.keys.size() - 1));
    const u32 a = static_cast<u32>(frame);
    const u32 b = std::min(a + 1, static_cast<u32>(clip.keys.size() - 1));
    const f32 u = frame - static_cast<f32>(a);
    const CameraKey& ka = clip.keys[a];
    const CameraKey& kb = clip.keys[b];
    return {lerp(ka.position, kb.position, u), nlerp(ka.rotation, kb.rotation, u), lerp(ka.fovDegrees, kb.fovDegrees, u)};
}

void Cutscenes::build(const sim::Table<CutsceneRow>& table, std::string* warnings) {
    m_cutscenes.clear();
    for (u32 i = 0; i < table.size(); ++i) {
        Cutscene c;
        c.name = table.id(i);
        c.row = table[i];
        // events: "seconds row seconds row ..."
        std::vector<std::string> words;
        for (size_t p = 0; p < c.row.events.size();) {
            while (p < c.row.events.size() && c.row.events[p] == ' ') ++p;
            const size_t start = p;
            while (p < c.row.events.size() && c.row.events[p] != ' ') ++p;
            if (p > start) words.push_back(c.row.events.substr(start, p - start));
        }
        if (words.size() % 2) warn(warnings, c.name + ": events need \"seconds row\" pairs");
        for (size_t w = 0; w + 1 < words.size(); w += 2) {
            f32 at = 0.0f;
            const auto r = std::from_chars(words[w].data(), words[w].data() + words[w].size(), at);
            if (r.ec != std::errc{}) {
                warn(warnings, c.name + ": bad event time '" + words[w] + "'");
                continue;
            }
            c.events.emplace_back(at, words[w + 1]);
        }
        std::stable_sort(c.events.begin(), c.events.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        std::string path = c.row.file.empty() ? std::string() : findAssetFile(c.row.file);
        if (path.empty() && !c.row.file.empty()) path = findAssetFile(c.row.file + kCookedCameraSuffix);
        std::vector<CameraClip> clips;
        std::string error;
        if (path.empty())
            warn(warnings, c.name + ": file '" + c.row.file + "' not found");
        else if (!loadCameraClips(path, clips, error))
            warn(warnings, c.name + ": " + c.row.file + ": " + error);
        else {
            const CameraClip* pick = &clips[0];
            if (!c.row.clip.empty()) {
                pick = nullptr;
                for (const CameraClip& clip : clips)
                    if (clip.name == c.row.clip) pick = &clip;
                if (!pick) warn(warnings, c.name + ": no animation '" + c.row.clip + "' in " + c.row.file);
            }
            if (pick) c.clip = std::make_shared<const CameraClip>(*pick);
        }
        m_cutscenes.push_back(std::move(c));
    }
}

bool Cutscenes::load() {
    m_path = findAsset(std::string("assets/") + kCutsceneTable);
    m_time = m_path.empty() ? 0 : fileWriteTime(m_path);
    sim::Table<CutsceneRow> table;
    if (!m_path.empty()) data::loadTable(table, kCutsceneTable);
    std::string warnings;
    build(table, &warnings);
    if (!warnings.empty()) MEALKIT_WARN("assets/%s: %s", kCutsceneTable, warnings.c_str());
    return !m_path.empty();
}

bool Cutscenes::poll(f64 dt) {
    m_check -= dt;
    if (m_check > 0.0) return false;
    m_check = kTableCheckSeconds;
    const std::string path = findAsset(std::string("assets/") + kCutsceneTable);
    const unsigned long long t = path.empty() ? 0 : fileWriteTime(path);
    if (path == m_path && t == m_time) return false;
    load();
    MEALKIT_INFO("Cutscenes: %u from %s", count(), path.empty() ? "(no cutscenes.table)" : path.c_str());
    return true;
}

const Cutscenes::Cutscene* Cutscenes::find(std::string_view name) const {
    for (const Cutscene& c : m_cutscenes)
        if (c.name == name) return &c;
    return nullptr;
}

} // namespace mealkit::asset
