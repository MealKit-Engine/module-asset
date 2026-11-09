#include "mealkit/asset/Cooked.h"

#include <cstring>
#include <type_traits>

namespace mealkit::asset {
namespace {

constexpr u32 kModelMagic = 0x444D4B4Du;  // "MKMD"
constexpr u32 kCameraMagic = 0x434D4B4Du; // "MKMC"
// Bumped when a layout below or a written struct changes (sizes are checked too, so a struct change without a bump still fails safely)
constexpr u32 kModelVersion = 1;
constexpr u32 kCameraVersion = 1;

struct Writer {
    std::string out;
    template <class T>
    void pod(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        out.append(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    template <class T>
    void array(const std::vector<T>& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        pod(static_cast<u32>(v.size()));
        if (!v.empty()) out.append(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
    }
    void text(const std::string& s) {
        pod(static_cast<u32>(s.size()));
        out += s;
    }
};

struct Reader {
    std::string_view in;
    size_t at = 0;
    bool ok = true;
    template <class T>
    void pod(T& v) {
        if (!ok || at + sizeof(T) > in.size()) {
            ok = false;
            return;
        }
        std::memcpy(&v, in.data() + at, sizeof(T));
        at += sizeof(T);
    }
    template <class T>
    void array(std::vector<T>& v) {
        u32 n = 0;
        pod(n);
        if (!ok || at + static_cast<size_t>(n) * sizeof(T) > in.size()) {
            ok = false;
            return;
        }
        v.resize(n);
        if (n) std::memcpy(v.data(), in.data() + at, static_cast<size_t>(n) * sizeof(T));
        at += static_cast<size_t>(n) * sizeof(T);
    }
    void text(std::string& s) {
        u32 n = 0;
        pod(n);
        if (!ok || at + n > in.size()) {
            ok = false;
            return;
        }
        s.assign(in.data() + at, n);
        at += n;
    }
};

// Sizes of the structs written as is: a layout change reads as "cook again"
constexpr u32 kLayout = static_cast<u32>(sizeof(Vertex) | sizeof(MaterialData) << 8 | sizeof(JointPose) << 16) ^
                        static_cast<u32>(sizeof(Mat4) << 24);

bool header(Reader& r, u32 magic, u32 version, std::string& error) {
    u32 m = 0, v = 0, layout = 0;
    r.pod(m);
    r.pod(v);
    r.pod(layout);
    if (!r.ok || m != magic) return error = "not a cooked file", false;
    if (v != version || layout != kLayout) return error = "cooked by another engine version (cook the game again)", false;
    return true;
}

} // namespace

std::string writeCookedModel(const CookedModel& m) {
    Writer w;
    w.pod(kModelMagic);
    w.pod(kModelVersion);
    w.pod(kLayout);
    w.pod(m.fileHash);
    w.pod(m.boundsMin);
    w.pod(m.boundsMax);
    w.array(m.hullPoints);
    w.pod(m.bones);
    w.pod(static_cast<u32>(m.skeleton.joints.size()));
    for (const Skeleton::Joint& j : m.skeleton.joints) {
        w.text(j.name);
        w.pod(j.parent);
        w.pod(j.rest);
        w.pod(j.inverseBind);
    }
    w.pod(m.skeleton.root);
    w.pod(static_cast<u32>(m.clips.size()));
    for (const AnimationClip& c : m.clips) {
        w.text(c.name);
        w.pod(c.frames);
        w.pod(c.duration);
        w.array(c.poses);
    }
    w.array(m.materials);
    w.pod(static_cast<u32>(m.textures.size()));
    for (const CompressedTexture& t : m.textures) {
        w.pod(t.width);
        w.pod(t.height);
        w.pod(t.mipLevels);
        w.pod(t.format);
        w.array(t.data);
    }
    w.pod(static_cast<u32>(m.lods.size()));
    for (const MeshData& mesh : m.lods) {
        w.array(mesh.vertices);
        w.array(mesh.indices);
        w.array(mesh.skin);
    }
    return std::move(w.out);
}

bool readCookedModel(std::string_view bytes, CookedModel& m, std::string& error) {
    Reader r{bytes};
    if (!header(r, kModelMagic, kModelVersion, error)) return false;
    m = {};
    r.pod(m.fileHash);
    r.pod(m.boundsMin);
    r.pod(m.boundsMax);
    r.array(m.hullPoints);
    r.pod(m.bones);
    u32 n = 0;
    r.pod(n);
    for (u32 i = 0; i < n && r.ok; ++i) {
        Skeleton::Joint j;
        r.text(j.name);
        r.pod(j.parent);
        r.pod(j.rest);
        r.pod(j.inverseBind);
        m.skeleton.joints.push_back(std::move(j));
    }
    r.pod(m.skeleton.root);
    r.pod(n);
    for (u32 i = 0; i < n && r.ok; ++i) {
        AnimationClip c;
        r.text(c.name);
        r.pod(c.frames);
        r.pod(c.duration);
        r.array(c.poses);
        m.clips.push_back(std::move(c));
    }
    r.array(m.materials);
    r.pod(n);
    for (u32 i = 0; i < n && r.ok; ++i) {
        CompressedTexture t;
        r.pod(t.width);
        r.pod(t.height);
        r.pod(t.mipLevels);
        r.pod(t.format);
        r.array(t.data);
        m.textures.push_back(std::move(t));
    }
    r.pod(n);
    for (u32 i = 0; i < n && r.ok; ++i) {
        MeshData mesh;
        r.array(mesh.vertices);
        r.array(mesh.indices);
        r.array(mesh.skin);
        m.lods.push_back(std::move(mesh));
    }
    if (!r.ok || r.at != bytes.size() || m.lods.empty()) return error = "damaged cooked model", false;
    return true;
}

std::string writeCameraClips(const std::vector<CameraClip>& clips) {
    Writer w;
    w.pod(kCameraMagic);
    w.pod(kCameraVersion);
    w.pod(kLayout);
    w.pod(static_cast<u32>(clips.size()));
    for (const CameraClip& c : clips) {
        w.text(c.name);
        w.pod(c.duration);
        w.array(c.keys);
    }
    return std::move(w.out);
}

bool readCameraClips(std::string_view bytes, std::vector<CameraClip>& out, std::string& error) {
    Reader r{bytes};
    if (!header(r, kCameraMagic, kCameraVersion, error)) return false;
    out.clear();
    u32 n = 0;
    r.pod(n);
    for (u32 i = 0; i < n && r.ok; ++i) {
        CameraClip c;
        r.text(c.name);
        r.pod(c.duration);
        r.array(c.keys);
        out.push_back(std::move(c));
    }
    if (!r.ok || r.at != bytes.size() || out.empty()) return error = "damaged cooked camera", false;
    return true;
}

} // namespace mealkit::asset
