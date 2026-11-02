#include "mealkit/asset/Environment.h"
#include "mealkit/asset/Cooked.h"

#include "mealkit/core/Hash.h"
#include "mealkit/core/Log.h"
#include "mealkit/core/Profile.h"
#include "mealkit/platform/Paths.h"
#include "mealkit/render/Renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

#if MEALKIT_DEBUG_UI
#include "stb_image.h"
#endif

namespace mealkit::asset {
namespace {

// Cache file: "MKEV", format version, sky face size, reflection face size, reflection mip count, 0, SH 27 floats, sky texels, reflection texels (RGB9E5)
constexpr u32 kMagic = 0x56454B4D; // "MKEV"
constexpr u32 kVersion = 1;        // bump when the baking method changes (it's in the key, so old files aren't used)
constexpr u32 kMaxSkySize = 512;
constexpr u32 kReflectSize = 128;
constexpr u32 kReflectMips = 6;   // 128 → 4
constexpr u32 kGgxSamples = 64;   // samples per texel per mip (filtered reads reduce noise)
constexpr f32 kRadianceClamp = 16.0f; // clamp very bright texels like the sun in reflections/ambient (the scene's sunlight handles the sun)
constexpr f32 kPi = 3.14159265358979f;

struct Header {
    u32 magic, version, skySize, reflectSize, reflectMips, reserved;
};

// Equirectangular mip pyramid (2×2 box, linear rgb)
struct Equirect {
    struct Level {
        u32 width = 0, height = 0;
        std::vector<f32> rgb;
    };
    std::vector<Level> levels;

    void build(const f32* rgb, u32 width, u32 height, f32 clampTo) {
        levels.clear();
        Level base;
        base.width = width;
        base.height = height;
        base.rgb.assign(rgb, rgb + static_cast<size_t>(width) * height * 3);
        for (f32& v : base.rgb) v = std::isfinite(v) ? std::clamp(v, 0.0f, clampTo) : 0.0f;
        levels.push_back(std::move(base));
        while (levels.back().width > 8 && levels.back().height > 4) {
            const Level& s = levels.back();
            Level d;
            d.width = s.width / 2;
            d.height = s.height / 2;
            d.rgb.resize(static_cast<size_t>(d.width) * d.height * 3);
            for (u32 y = 0; y < d.height; ++y)
                for (u32 x = 0; x < d.width; ++x)
                    for (u32 c = 0; c < 3; ++c) {
                        f32 sum = 0.0f;
                        for (u32 dy = 0; dy < 2; ++dy)
                            for (u32 dx = 0; dx < 2; ++dx) sum += s.rgb[((y * 2 + dy) * s.width + x * 2 + dx) * 3 + c];
                        d.rgb[(y * d.width + x) * 3 + c] = sum * 0.25f;
                    }
            levels.push_back(std::move(d));
        }
    }

    // direction → (u, v): u is azimuth (+z forward at center, increasing toward +x right), v from top (0) to bottom (1)
    static void uv(Vec3 d, f32& u, f32& v) {
        u = 0.5f + std::atan2(d.x, d.z) / (2.0f * kPi);
        v = std::acos(std::clamp(d.y, -1.0f, 1.0f)) / kPi;
    }

    Vec3 bilinear(const Level& l, f32 u, f32 v) const {
        const f32 x = u * static_cast<f32>(l.width) - 0.5f, y = v * static_cast<f32>(l.height) - 0.5f;
        const f32 fx = std::floor(x), fy = std::floor(y);
        const f32 tx = x - fx, ty = y - fy;
        auto at = [&](i32 px, i32 py) {
            const i32 w = static_cast<i32>(l.width), h = static_cast<i32>(l.height);
            px = ((px % w) + w) % w; // wraps horizontally
            py = std::clamp(py, 0, h - 1);
            const f32* p = &l.rgb[(static_cast<size_t>(py) * l.width + static_cast<size_t>(px)) * 3];
            return Vec3{p[0], p[1], p[2]};
        };
        const i32 ix = static_cast<i32>(fx), iy = static_cast<i32>(fy);
        const Vec3 a = lerp(at(ix, iy), at(ix + 1, iy), tx), b = lerp(at(ix, iy + 1), at(ix + 1, iy + 1), tx);
        return lerp(a, b, ty);
    }

    Vec3 sample(Vec3 d, f32 lod) const {
        f32 u, v;
        uv(d, u, v);
        lod = std::clamp(lod, 0.0f, static_cast<f32>(levels.size() - 1));
        const u32 l0 = static_cast<u32>(lod), l1 = std::min<u32>(l0 + 1, static_cast<u32>(levels.size() - 1));
        return lerp(bilinear(levels[l0], u, v), bilinear(levels[l1], u, v), lod - static_cast<f32>(l0));
    }

    // Average solid angle of one texel at pyramid level 0
    f32 texelSolidAngle() const { return 4.0f * kPi / (static_cast<f32>(levels[0].width) * static_cast<f32>(levels[0].height)); }
};

f32 lodFor(const Equirect& e, f32 solidAngle) { return std::max(0.0f, 0.5f * std::log2(solidAngle / e.texelSolidAngle())); }

// Real spherical harmonics orders 0~2 (standard constants with axes renamed for y-up). Same as evalSh in the Forward.hlsl shader
void shBasis(Vec3 n, f32 (&y)[9]) {
    y[0] = 0.282095f;
    y[1] = 0.488603f * n.y;
    y[2] = 0.488603f * n.z;
    y[3] = 0.488603f * n.x;
    y[4] = 1.092548f * n.x * n.y;
    y[5] = 1.092548f * n.y * n.z;
    y[6] = 0.315392f * (3.0f * n.z * n.z - 1.0f);
    y[7] = 1.092548f * n.x * n.z;
    y[8] = 0.546274f * (n.x * n.x - n.y * n.y);
}

Vec3 hammersleyGgx(u32 i, u32 count, f32 a, Vec3 n) {
    u32 bits = i;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    const f32 x1 = static_cast<f32>(i) / static_cast<f32>(count), x2 = static_cast<f32>(bits) * 2.3283064365386963e-10f;
    const f32 phi = 2.0f * kPi * x1;
    const f32 cosTheta = std::sqrt((1.0f - x2) / (1.0f + (a * a - 1.0f) * x2));
    const f32 sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
    const Vec3 up = std::abs(n.y) < 0.999f ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    const Vec3 tx = normalize(cross(up, n)), ty = cross(n, tx);
    return normalize(tx * (sinTheta * std::cos(phi)) + ty * (sinTheta * std::sin(phi)) + n * cosTheta);
}

void faceTexel(u32 size, u32 x, u32 y, f32& s, f32& t) {
    s = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(size) * 2.0f - 1.0f;
    t = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(size) * 2.0f - 1.0f;
}
f32 cubeTexelAngle(u32 size) { return 4.0f * kPi / (6.0f * static_cast<f32>(size) * static_cast<f32>(size)); }

// Reflection cube (appended to out per face starting from mip 0): mip k = roughness k/(mip count-1). 0 is a mirror (filtered read by texel size),
// the rest are GGX importance samples (assuming N = V = R). sample(direction, source level), lodFor(solid angle) → source level. Shared by sky HDRI and reflection probes
template <class Sample, class LodFor>
void prefilterReflect(const Sample& sample, const LodFor& lodFor, u32 reflectSize, u32 reflectMips, std::vector<u32>& out) {
    out.clear();
    for (u32 face = 0; face < 6; ++face)
        for (u32 mip = 0; mip < reflectMips; ++mip) {
            const u32 size = std::max(1u, reflectSize >> mip);
            const f32 roughness = static_cast<f32>(mip) / static_cast<f32>(reflectMips - 1);
            const f32 a = std::max(roughness * roughness, 1e-4f);
            for (u32 y = 0; y < size; ++y)
                for (u32 x = 0; x < size; ++x) {
                    f32 s, t;
                    faceTexel(size, x, y, s, t);
                    const Vec3 n = cubeDirection(face, s, t);
                    Vec3 c{};
                    if (mip == 0) {
                        c = sample(n, lodFor(cubeTexelAngle(size)));
                    } else {
                        f32 weight = 0.0f;
                        for (u32 i = 0; i < kGgxSamples; ++i) {
                            const Vec3 h = hammersleyGgx(i, kGgxSamples, a, n);
                            const f32 ndh = std::max(dot(n, h), 0.0f);
                            const Vec3 l = h * (2.0f * ndh) - n;
                            const f32 ndl = dot(n, l);
                            if (ndl <= 0.0f) continue;
                            const f32 dd = ndh * ndh * (a * a - 1.0f) + 1.0f;
                            const f32 d = a * a / (kPi * dd * dd);
                            const f32 pdf = std::max(d * 0.25f, 1e-6f); // D·NdotH / (4·VdotH), N = V
                            c = c + sample(l, lodFor(1.0f / (static_cast<f32>(kGgxSamples) * pdf)) + 1.0f) * ndl;
                            weight += ndl;
                        }
                        if (weight > 0.0f) c = c * (1.0f / weight);
                    }
                    out.push_back(packRgb9e5(c.x, c.y, c.z));
                }
        }
}

// Mip pyramid of the six cube faces (2×2 box, linear rgb). Bilinear reads within a face only (face edges are clamped to edge texels)
struct CubeSource {
    std::vector<u32> sizes;                 // face size per level
    std::vector<std::vector<f32>> levels;   // per level 6 faces × size² × 3

    void build(const f32* faces, u32 size) {
        sizes = {size};
        levels.assign(1, std::vector<f32>(faces, faces + static_cast<size_t>(size) * size * 18));
        for (f32& v : levels[0]) v = std::isfinite(v) ? std::clamp(v, 0.0f, kRadianceClamp) : 0.0f;
        while (sizes.back() > 1) {
            const u32 s = sizes.back(), d = s / 2;
            const std::vector<f32>& src = levels.back();
            std::vector<f32> dst(static_cast<size_t>(d) * d * 18);
            for (u32 face = 0; face < 6; ++face)
                for (u32 y = 0; y < d; ++y)
                    for (u32 x = 0; x < d; ++x)
                        for (u32 c = 0; c < 3; ++c) {
                            f32 sum = 0.0f;
                            for (u32 dy = 0; dy < 2; ++dy)
                                for (u32 dx = 0; dx < 2; ++dx) sum += src[((static_cast<size_t>(face) * s + y * 2 + dy) * s + x * 2 + dx) * 3 + c];
                            dst[((static_cast<size_t>(face) * d + y) * d + x) * 3 + c] = sum * 0.25f;
                        }
            sizes.push_back(d);
            levels.push_back(std::move(dst));
        }
    }

    // direction → face and s/t (-1..1, inverse of cubeDirection)
    static void faceOf(Vec3 d, u32& face, f32& s, f32& t) {
        const f32 ax = std::abs(d.x), ay = std::abs(d.y), az = std::abs(d.z);
        if (ax >= ay && ax >= az) {
            face = d.x > 0.0f ? 0 : 1;
            s = (d.x > 0.0f ? -d.z : d.z) / ax;
            t = -d.y / ax;
        } else if (ay >= az) {
            face = d.y > 0.0f ? 2 : 3;
            s = d.x / ay;
            t = (d.y > 0.0f ? d.z : -d.z) / ay;
        } else {
            face = d.z > 0.0f ? 4 : 5;
            s = (d.z > 0.0f ? d.x : -d.x) / az;
            t = -d.y / az;
        }
    }

    Vec3 bilinear(u32 level, u32 face, f32 s, f32 t) const {
        const u32 size = sizes[level];
        const std::vector<f32>& l = levels[level];
        const f32 x = (s * 0.5f + 0.5f) * static_cast<f32>(size) - 0.5f, y = (t * 0.5f + 0.5f) * static_cast<f32>(size) - 0.5f;
        const f32 fx = std::floor(x), fy = std::floor(y);
        const f32 tx = x - fx, ty = y - fy;
        auto at = [&](i32 px, i32 py) {
            px = std::clamp(px, 0, static_cast<i32>(size) - 1);
            py = std::clamp(py, 0, static_cast<i32>(size) - 1);
            const f32* p = &l[((static_cast<size_t>(face) * size + static_cast<size_t>(py)) * size + static_cast<size_t>(px)) * 3];
            return Vec3{p[0], p[1], p[2]};
        };
        const i32 ix = static_cast<i32>(fx), iy = static_cast<i32>(fy);
        return lerp(lerp(at(ix, iy), at(ix + 1, iy), tx), lerp(at(ix, iy + 1), at(ix + 1, iy + 1), tx), ty);
    }

    Vec3 sample(Vec3 d, f32 lod) const {
        u32 face;
        f32 s, t;
        faceOf(d, face, s, t);
        lod = std::clamp(lod, 0.0f, static_cast<f32>(levels.size() - 1));
        const u32 l0 = static_cast<u32>(lod), l1 = std::min<u32>(l0 + 1, static_cast<u32>(levels.size() - 1));
        return lerp(bilinear(l0, face, s, t), bilinear(l1, face, s, t), lod - static_cast<f32>(l0));
    }

    f32 lodFor(f32 solidAngle) const { return std::max(0.0f, 0.5f * std::log2(solidAngle / cubeTexelAngle(sizes[0]))); }
};

std::string cacheName(const std::string& file) {
    Hasher h;
    h.bytes(file.data(), file.size());
    h.add(static_cast<u64>(kVersion));
    char name[40];
    std::snprintf(name, sizeof(name), "%016llx.mkenv", static_cast<unsigned long long>(h.value));
    return name;
}

bool readCache(const std::string& path, EnvironmentMap& out) {
    std::string file;
    if (!readTextFile(path, file) || file.size() < sizeof(Header) + sizeof(out.sh)) return false;
    Header hd;
    std::memcpy(&hd, file.data(), sizeof(hd));
    if (hd.magic != kMagic || hd.version != kVersion || hd.skySize == 0 || hd.skySize > 4096 || hd.reflectSize == 0 ||
        hd.reflectMips == 0 || hd.reflectMips > 16)
        return false;
    size_t reflectTexels = 0;
    for (u32 mip = 0; mip < hd.reflectMips; ++mip) reflectTexels += static_cast<size_t>(std::max(1u, hd.reflectSize >> mip)) * std::max(1u, hd.reflectSize >> mip);
    const size_t skyTexels = static_cast<size_t>(hd.skySize) * hd.skySize * 6;
    reflectTexels *= 6;
    if (file.size() != sizeof(Header) + sizeof(out.sh) + (skyTexels + reflectTexels) * 4) return false;
    const char* p = file.data() + sizeof(Header);
    std::memcpy(out.sh, p, sizeof(out.sh));
    p += sizeof(out.sh);
    out.skySize = hd.skySize;
    out.sky.resize(skyTexels);
    std::memcpy(out.sky.data(), p, skyTexels * 4);
    p += skyTexels * 4;
    out.reflectSize = hd.reflectSize;
    out.reflectMips = hd.reflectMips;
    out.reflect.resize(reflectTexels);
    std::memcpy(out.reflect.data(), p, reflectTexels * 4);
    return true;
}

} // namespace

u32 packRgb9e5(f32 r, f32 g, f32 b) {
    static constexpr f32 kMax = 65408.0f; // (511 / 512) × 2^16
    auto fix = [](f32 v) { return std::isfinite(v) ? std::clamp(v, 0.0f, kMax) : 0.0f; };
    r = fix(r), g = fix(g), b = fix(b);
    const f32 maxc = std::max({r, g, b});
    if (maxc <= 0.0f) return 0;
    i32 exp = std::max(-16, static_cast<i32>(std::floor(std::log2(maxc)))) + 1 + 15;
    f32 denom = std::exp2(static_cast<f32>(exp - 15 - 9));
    if (static_cast<i32>(std::floor(maxc / denom + 0.5f)) == 512) {
        denom *= 2.0f;
        ++exp;
    }
    const u32 rm = static_cast<u32>(std::floor(r / denom + 0.5f)), gm = static_cast<u32>(std::floor(g / denom + 0.5f)),
              bm = static_cast<u32>(std::floor(b / denom + 0.5f));
    return std::min(rm, 511u) | (std::min(gm, 511u) << 9) | (std::min(bm, 511u) << 18) | (static_cast<u32>(exp) << 27);
}

void unpackRgb9e5(u32 v, f32 out[3]) {
    const f32 scale = std::exp2(static_cast<f32>(static_cast<i32>(v >> 27) - 15 - 9));
    out[0] = static_cast<f32>(v & 511u) * scale;
    out[1] = static_cast<f32>((v >> 9) & 511u) * scale;
    out[2] = static_cast<f32>((v >> 18) & 511u) * scale;
}

Vec3 cubeDirection(u32 face, f32 s, f32 t) {
    Vec3 d;
    switch (face) {
    case 0: d = {1.0f, -t, -s}; break;   // +X
    case 1: d = {-1.0f, -t, s}; break;   // -X
    case 2: d = {s, 1.0f, t}; break;     // +Y
    case 3: d = {s, -1.0f, -t}; break;   // -Y
    case 4: d = {s, -t, 1.0f}; break;    // +Z
    default: d = {-s, -t, -1.0f}; break; // -Z
    }
    return normalize(d);
}

Vec3 evalSh(const f32 (&sh)[9][3], Vec3 n) {
    f32 y[9];
    shBasis(n, y);
    Vec3 c{};
    for (u32 i = 0; i < 9; ++i) c = c + Vec3{sh[i][0], sh[i][1], sh[i][2]} * y[i];
    return Vec3{std::max(c.x, 0.0f), std::max(c.y, 0.0f), std::max(c.z, 0.0f)};
}

void bakeEnvironment(const f32* rgb, u32 width, u32 height, EnvironmentMap& out) {
    MEALKIT_ZONE("Environment bake");
    Equirect full, clamped;
    full.build(rgb, width, height, 65408.0f);
    clamped.build(rgb, width, height, kRadianceClamp);

    // Sky cube: face size = width/4 (max 512), read from the level matching the texel solid angle
    out.skySize = std::clamp<u32>(width / 4, 16, kMaxSkySize);
    out.sky.resize(static_cast<size_t>(out.skySize) * out.skySize * 6);
    const f32 skyLod = lodFor(full, cubeTexelAngle(out.skySize));
    for (u32 face = 0; face < 6; ++face)
        for (u32 y = 0; y < out.skySize; ++y)
            for (u32 x = 0; x < out.skySize; ++x) {
                f32 s, t;
                faceTexel(out.skySize, x, y, s, t);
                const Vec3 c = full.sample(cubeDirection(face, s, t), skyLod);
                out.sky[(static_cast<size_t>(face) * out.skySize + y) * out.skySize + x] = packRgb9e5(c.x, c.y, c.z);
            }

    // Reflection cube (mip = roughness)
    out.reflectSize = kReflectSize;
    out.reflectMips = kReflectMips;
    prefilterReflect([&](Vec3 d, f32 lod) { return clamped.sample(d, lod); },
                     [&](f32 solidAngle) { return lodFor(clamped, solidAngle); }, kReflectSize, kReflectMips, out.reflect);

    // Ambient SH: integrate per texel of a small level (width ≤ 128) by solid angle, and multiply in the Lambert convolution (A_l / π)
    const Equirect::Level& l = clamped.levels[std::min<size_t>(clamped.levels.size() - 1, static_cast<size_t>(std::max(0.0f, std::log2(static_cast<f32>(width) / 128.0f))))];
    std::memset(out.sh, 0, sizeof(out.sh));
    for (u32 y = 0; y < l.height; ++y) {
        const f32 theta = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(l.height) * kPi;
        const f32 solid = (2.0f * kPi / static_cast<f32>(l.width)) * (kPi / static_cast<f32>(l.height)) * std::sin(theta);
        for (u32 x = 0; x < l.width; ++x) {
            const f32 phi = ((static_cast<f32>(x) + 0.5f) / static_cast<f32>(l.width) - 0.5f) * 2.0f * kPi;
            const Vec3 d{std::sin(theta) * std::sin(phi), std::cos(theta), std::sin(theta) * std::cos(phi)}; // inverse of uv()
            f32 basis[9];
            shBasis(d, basis);
            const f32* p = &l.rgb[(static_cast<size_t>(y) * l.width + x) * 3];
            for (u32 i = 0; i < 9; ++i)
                for (u32 c = 0; c < 3; ++c) out.sh[i][c] += p[c] * basis[i] * solid;
        }
    }
    const f32 band[9] = {1.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f};
    for (u32 i = 0; i < 9; ++i)
        for (u32 c = 0; c < 3; ++c) out.sh[i][c] *= band[i];
}

namespace {
constexpr u32 kProbeMagic = 0x52504B4D; // "MKPR"
constexpr u32 kProbeVersion = 1;
struct ProbeHeader {
    u32 magic, version, size, mips, reserved[2];
};
size_t probeTexels(u32 size, u32 mips) {
    size_t n = 0;
    for (u32 mip = 0; mip < mips; ++mip) n += static_cast<size_t>(std::max(1u, size >> mip)) * std::max(1u, size >> mip);
    return n * 6;
}
} // namespace

void bakeProbe(const f32* faces, u32 faceSize, ProbeMap& out) {
    MEALKIT_ZONE("Probe bake");
    CubeSource source;
    source.build(faces, faceSize);
    out.size = faceSize;
    out.mips = 1;
    while ((faceSize >> out.mips) >= 4 && out.mips < kReflectMips) ++out.mips; // 128 → 6 mips (down to 4), same roughness spacing as sky reflection
    prefilterReflect([&](Vec3 d, f32 lod) { return source.sample(d, lod); }, [&](f32 solidAngle) { return source.lodFor(solidAngle); },
                     out.size, out.mips, out.reflect);
}

bool saveProbe(const std::string& path, const ProbeMap& probe) {
    std::string data(sizeof(ProbeHeader) + probe.reflect.size() * 4, '\0');
    const ProbeHeader hd{kProbeMagic, kProbeVersion, probe.size, probe.mips, {0, 0}};
    std::memcpy(data.data(), &hd, sizeof(hd));
    std::memcpy(data.data() + sizeof(hd), probe.reflect.data(), probe.reflect.size() * 4);
    const size_t slash = path.find_last_of("/\\");
    if (slash != std::string::npos) createDirectories(path.substr(0, slash));
    return writeTextFile(path, data);
}

bool loadProbe(const std::string& path, ProbeMap& out, std::string& error) {
    std::string file;
    if (!readTextFile(path, file)) {
        error = "file not found (bake the probe in the editor)";
        return false;
    }
    ProbeHeader hd{};
    if (file.size() >= sizeof(hd)) std::memcpy(&hd, file.data(), sizeof(hd));
    if (hd.magic != kProbeMagic || hd.version != kProbeVersion || hd.size == 0 || hd.size > 1024 || hd.mips == 0 || hd.mips > 16 ||
        file.size() != sizeof(hd) + probeTexels(hd.size, hd.mips) * 4) {
        error = "not a probe file of this engine version (bake it again)";
        return false;
    }
    out.size = hd.size;
    out.mips = hd.mips;
    out.reflect.resize(probeTexels(hd.size, hd.mips));
    std::memcpy(out.reflect.data(), file.data() + sizeof(hd), out.reflect.size() * 4);
    return true;
}

// Files with that extension anywhere under the kind folder kind/ (any subfolders), paths relative to assets
static std::vector<std::string> listKindFiles(const std::string& assetsRoot, const char* kind, const char* extension) {
    std::vector<std::string> out;
    std::vector<std::string> folders{kind};
    while (!folders.empty()) {
        const std::string folder = folders.back();
        folders.pop_back();
        for (const std::string& f : listFiles(assetsRoot + "/" + folder, extension)) out.push_back(folder + "/" + f);
        for (const std::string& d : listDirectories(assetsRoot + "/" + folder)) folders.push_back(folder + "/" + d);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::string> listHdris(const std::string& assetsRoot) { return listKindFiles(assetsRoot, "textures", ".hdr"); }
std::vector<std::string> listLuts(const std::string& assetsRoot) { return listKindFiles(assetsRoot, "lut", ".cube"); }

bool parseCubeLut(const std::string& text, std::vector<f32>& out, std::string& error) {
    u32 size = 0;
    f32 domainMin[3] = {0, 0, 0}, domainMax[3] = {1, 1, 1};
    std::vector<f32> cells;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t end = text.find('\n', pos);
        if (end == std::string::npos) end = text.size();
        std::string line = text.substr(pos, end - pos);
        pos = end + 1;
        if (const size_t hash = line.find('#'); hash != std::string::npos) line.resize(hash);
        const char* s = line.c_str();
        while (*s == ' ' || *s == '\t') ++s;
        if (!*s || *s == '\r') continue;
        f32 v[3];
        if (std::strncmp(s, "LUT_3D_SIZE", 11) == 0) {
            size = static_cast<u32>(std::atoi(s + 11));
        } else if (std::strncmp(s, "LUT_1D_SIZE", 11) == 0) {
            error = "1D LUTs are not supported (export a 3D LUT)";
            return false;
        } else if (std::strncmp(s, "DOMAIN_MIN", 10) == 0) {
            if (std::sscanf(s + 10, "%f %f %f", &domainMin[0], &domainMin[1], &domainMin[2]) != 3) return error = "bad DOMAIN_MIN", false;
        } else if (std::strncmp(s, "DOMAIN_MAX", 10) == 0) {
            if (std::sscanf(s + 10, "%f %f %f", &domainMax[0], &domainMax[1], &domainMax[2]) != 3) return error = "bad DOMAIN_MAX", false;
        } else if ((*s >= '0' && *s <= '9') || *s == '-' || *s == '.') {
            if (std::sscanf(s, "%f %f %f", &v[0], &v[1], &v[2]) != 3) return error = "bad table line", false;
            cells.insert(cells.end(), v, v + 3);
        } // skip other keys like TITLE
    }
    if (size < 2 || size > 256) return error = "missing or invalid LUT_3D_SIZE", false;
    if (cells.size() != static_cast<size_t>(size) * size * size * 3)
        return error = "table has " + std::to_string(cells.size() / 3) + " entries, expected " + std::to_string(size * size * size), false;

    // Trilinear interpolation over the source grid (input 0..1 → source domain → grid coordinates)
    auto fetch = [&](u32 r, u32 g, u32 b, u32 c) { return cells[((static_cast<size_t>(b) * size + g) * size + r) * 3 + c]; };
    constexpr u32 n = render::kLutSize;
    out.resize(static_cast<size_t>(n) * n * n * 3);
    for (u32 b = 0; b < n; ++b)
        for (u32 g = 0; g < n; ++g)
            for (u32 r = 0; r < n; ++r) {
                const u32 in[3] = {r, g, b};
                f32 x[3];
                u32 i0[3], i1[3];
                f32 t[3];
                for (u32 k = 0; k < 3; ++k) {
                    const f32 u = static_cast<f32>(in[k]) / (n - 1);
                    const f32 span = domainMax[k] - domainMin[k];
                    x[k] = std::clamp(span > 0.0f ? (u - domainMin[k]) / span : u, 0.0f, 1.0f) * static_cast<f32>(size - 1);
                    i0[k] = std::min(static_cast<u32>(x[k]), size - 2);
                    i1[k] = i0[k] + 1;
                    t[k] = x[k] - static_cast<f32>(i0[k]);
                }
                f32* dst = &out[((static_cast<size_t>(b) * n + g) * n + r) * 3];
                for (u32 c = 0; c < 3; ++c) {
                    auto lerp1 = [](f32 a, f32 bb, f32 f) { return a + (bb - a) * f; };
                    const f32 c00 = lerp1(fetch(i0[0], i0[1], i0[2], c), fetch(i1[0], i0[1], i0[2], c), t[0]);
                    const f32 c10 = lerp1(fetch(i0[0], i1[1], i0[2], c), fetch(i1[0], i1[1], i0[2], c), t[0]);
                    const f32 c01 = lerp1(fetch(i0[0], i0[1], i1[2], c), fetch(i1[0], i0[1], i1[2], c), t[0]);
                    const f32 c11 = lerp1(fetch(i0[0], i1[1], i1[2], c), fetch(i1[0], i1[1], i1[2], c), t[0]);
                    dst[c] = std::clamp(lerp1(lerp1(c00, c10, t[1]), lerp1(c01, c11, t[1]), t[2]), 0.0f, 1.0f);
                }
            }
    return true;
}

bool loadCubeLut(const std::string& path, std::vector<f32>& out, std::string& error) {
    std::string text;
    if (!readTextFile(path, text)) return error = "file not found", false;
    return parseCubeLut(text, out, error);
}

bool saveNeutralCubeLut(const std::string& path) {
    constexpr u32 n = render::kLutSize;
    std::string text = "# MealKit neutral LUT (grade the capture next to this file, then export a .cube)\nTITLE \"MealKit neutral\"\nLUT_3D_SIZE 32\n";
    char line[64];
    for (u32 b = 0; b < n; ++b)
        for (u32 g = 0; g < n; ++g)
            for (u32 r = 0; r < n; ++r) {
                std::snprintf(line, sizeof(line), "%.6f %.6f %.6f\n", static_cast<f32>(r) / (n - 1), static_cast<f32>(g) / (n - 1),
                              static_cast<f32>(b) / (n - 1));
                text += line;
            }
    return writeTextFile(path, text);
}

std::string environmentCacheFolder() { return projectRoot() + "/build/cache/environments"; }

std::string environmentCacheFile(const std::string& hdrPath) {
    std::string file;
    if (!readTextFile(hdrPath, file)) return {};
    const std::string written = environmentCacheFolder() + "/" + cacheName(file);
    return fileExists(written) ? written : std::string();
}

bool loadEnvironment(const std::string& hdrPath, EnvironmentMap& out, std::string& error) {
    MEALKIT_ZONE("Environment cache");
    const std::string_view cooked = kCookedEnvironmentSuffix;
    if (hdrPath.size() > cooked.size() && hdrPath.compare(hdrPath.size() - cooked.size(), cooked.size(), cooked) == 0) {
        if (readCache(hdrPath, out)) return true; // cooked into the pack: the cache file as is
        error = "cooked environment of another engine version (cook the game again)";
        return false;
    }
    std::string file;
    if (!readTextFile(hdrPath, file)) {
        error = "file not found";
        return false;
    }
    const std::string name = cacheName(file);
    const std::string written = environmentCacheFolder() + "/" + name;
    if (readCache(written, out) || readCache(executableDirectory() + "/cache/environments/" + name, out)) return true;

#if MEALKIT_DEBUG_UI
    int w = 0, h = 0, channels = 0;
    f32* rgb = stbi_loadf_from_memory(reinterpret_cast<const stbi_uc*>(file.data()), static_cast<int>(file.size()), &w, &h, &channels, 3);
    if (!rgb) {
        error = std::string("cannot decode HDR image: ") + stbi_failure_reason();
        return false;
    }
    bakeEnvironment(rgb, static_cast<u32>(w), static_cast<u32>(h), out);
    stbi_image_free(rgb);

    std::string data(sizeof(Header) + sizeof(out.sh) + (out.sky.size() + out.reflect.size()) * 4, '\0');
    const Header hd{kMagic, kVersion, out.skySize, out.reflectSize, out.reflectMips, 0};
    char* p = data.data();
    std::memcpy(p, &hd, sizeof(hd));
    p += sizeof(hd);
    std::memcpy(p, out.sh, sizeof(out.sh));
    p += sizeof(out.sh);
    std::memcpy(p, out.sky.data(), out.sky.size() * 4);
    p += out.sky.size() * 4;
    std::memcpy(p, out.reflect.data(), out.reflect.size() * 4);
    if (!createDirectories(environmentCacheFolder()) || !writeTextFile(written, data))
        MEALKIT_WARN("Environment cache: cannot write %s", written.c_str());
    MEALKIT_INFO("Environment cache: baked %dx%d HDRI (sky %u, reflection %u, %u mips) -> %s", w, h, out.skySize,
                 out.reflectSize, out.reflectMips, name.c_str());
    return true;
#else
    error = "environment " + name + " is not in the environment cache (export the game again)";
    return false;
#endif
}

} // namespace mealkit::asset
