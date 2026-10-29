#include "mealkit/asset/TextureCache.h"

#include "mealkit/core/Hash.h"
#include "mealkit/core/Log.h"
#include "mealkit/core/Profile.h"
#include "mealkit/platform/Paths.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "stb_image.h"
#if MEALKIT_DEBUG_UI
#include "bc7enc.h"
#include "rgbcx.h"
#endif

namespace mealkit::asset {
namespace {

// Cache file: "MKTX", format version, platform format family (0 = BC, ASTC for mobile in stage 5), rhi::Format, size, mip count, data
constexpr u32 kMagic = 0x58544B4D; // "MKTX"
constexpr u32 kVersion = 1;        // bump when the compression method changes (it's in the key, so old files aren't used)
constexpr u32 kFamilyBc = 0;

struct Header {
    u32 magic, version, family, format, width, height, mipLevels, reserved;
};

std::string cacheName(const void* encoded, size_t bytes, TextureUsage usage) {
    Hasher h;
    h.bytes(encoded, bytes);
    h.add(static_cast<u64>(usage));
    h.add(static_cast<u64>(kVersion));
    char name[40];
    std::snprintf(name, sizeof(name), "%016llx.mktex", static_cast<unsigned long long>(h.value));
    return name;
}

bool readCache(const std::string& path, CompressedTexture& out) {
    std::string file;
    if (!readTextFile(path, file) || file.size() < sizeof(Header)) return false;
    Header hd;
    std::memcpy(&hd, file.data(), sizeof(hd));
    if (hd.magic != kMagic || hd.version != kVersion || hd.family != kFamilyBc || hd.mipLevels == 0 || hd.mipLevels > 16)
        return false;
    out.width = hd.width;
    out.height = hd.height;
    out.mipLevels = hd.mipLevels;
    out.format = static_cast<rhi::Format>(hd.format);
    u64 expect = 0;
    for (u32 mip = 0; mip < hd.mipLevels; ++mip)
        expect += rhi::textureMipBytes(out.format, std::max(1u, hd.width >> mip), std::max(1u, hd.height >> mip));
    if (file.size() != sizeof(Header) + expect) return false;
    out.data.assign(file.begin() + sizeof(Header), file.end());
    return true;
}

f32 toLinear(u8 c) {
    const f32 v = c / 255.0f;
    return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
}
u8 toSrgb(f32 v) {
    v = std::clamp(v, 0.0f, 1.0f);
    const f32 s = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    return static_cast<u8>(s * 255.0f + 0.5f);
}

#if MEALKIT_DEBUG_UI
// Compress per 4×4 block. Edges repeat the last pixel
void compress(const std::vector<u8>& rgba, u32 w, u32 h, TextureUsage usage, std::vector<u8>& out) {
    static const bool ready = [] {
        bc7enc_compress_block_init();
        rgbcx::init();
        return true;
    }();
    (void)ready;
    bc7enc_compress_block_params params;
    bc7enc_compress_block_params_init(&params);
    if (usage != TextureUsage::Color) bc7enc_compress_block_params_init_linear_weights(&params);
    params.m_uber_level = 1; // between quality and speed (once at load)
    const u32 bw = (w + 3) / 4, bh = (h + 3) / 4;
    const size_t start = out.size();
    out.resize(start + static_cast<size_t>(bw) * bh * 16);
    u8 block[16 * 4];
    for (u32 by = 0; by < bh; ++by) {
        for (u32 bx = 0; bx < bw; ++bx) {
            for (u32 y = 0; y < 4; ++y)
                for (u32 x = 0; x < 4; ++x) {
                    const u32 sx = std::min(bx * 4 + x, w - 1), sy = std::min(by * 4 + y, h - 1);
                    std::memcpy(block + (y * 4 + x) * 4, &rgba[(static_cast<size_t>(sy) * w + sx) * 4], 4);
                }
            u8* dst = out.data() + start + (static_cast<size_t>(by) * bw + bx) * 16;
            if (usage == TextureUsage::Normal) rgbcx::encode_bc5_hq(dst, block, 0, 1, 4);
            else bc7enc_compress_block(dst, block, &params);
        }
    }
}
#endif

} // namespace

std::string textureCacheFolder() { return projectRoot() + "/build/cache/textures"; }

void buildMips(const u8* rgba, u32 width, u32 height, bool srgb, bool normal, std::vector<std::vector<u8>>& mips) {
    mips.clear();
    mips.emplace_back(rgba, rgba + static_cast<size_t>(width) * height * 4);
    u32 w = width, h = height;
    while (w > 1 || h > 1) {
        const u32 nw = std::max(1u, w / 2), nh = std::max(1u, h / 2);
        const std::vector<u8>& src = mips.back();
        std::vector<u8> dst(static_cast<size_t>(nw) * nh * 4);
        for (u32 y = 0; y < nh; ++y)
            for (u32 x = 0; x < nw; ++x) {
                f32 sum[4] = {};
                for (u32 dy = 0; dy < 2; ++dy)
                    for (u32 dx = 0; dx < 2; ++dx) {
                        const u32 sx = std::min(x * 2 + dx, w - 1), sy = std::min(y * 2 + dy, h - 1);
                        const u8* p = &src[(static_cast<size_t>(sy) * w + sx) * 4];
                        for (int c = 0; c < 4; ++c) sum[c] += (srgb && c < 3) ? toLinear(p[c]) : p[c] / 255.0f;
                    }
                u8* q = &dst[(static_cast<size_t>(y) * nw + x) * 4];
                if (normal) { // normals are averaged and renormalized to length 1
                    f32 n[3];
                    for (int c = 0; c < 3; ++c) n[c] = sum[c] / 2.0f - 1.0f; // (sum/4)*2-1
                    const f32 len = std::max(1e-6f, std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]));
                    for (int c = 0; c < 3; ++c) q[c] = static_cast<u8>(std::clamp(n[c] / len * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f + 0.5f);
                    q[3] = static_cast<u8>(sum[3] / 4.0f * 255.0f + 0.5f);
                    continue;
                }
                for (int c = 0; c < 4; ++c)
                    q[c] = (srgb && c < 3) ? toSrgb(sum[c] / 4.0f) : static_cast<u8>(sum[c] / 4.0f * 255.0f + 0.5f);
            }
        mips.push_back(std::move(dst));
        w = nw;
        h = nh;
    }
}

bool loadCompressedTexture(const void* encoded, size_t bytes, TextureUsage usage, CompressedTexture& out, std::string& error) {
    MEALKIT_ZONE("Texture cache");
    const std::string name = cacheName(encoded, bytes, usage);
    const std::string written = textureCacheFolder() + "/" + name;
    if (readCache(written, out) || readCache(executableDirectory() + "/cache/textures/" + name, out)) return true;

#if MEALKIT_DEBUG_UI
    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(static_cast<const stbi_uc*>(encoded), static_cast<int>(bytes), &w, &h, &channels, 4);
    if (!pixels) {
        error = std::string("cannot decode image: ") + stbi_failure_reason();
        return false;
    }
    std::vector<std::vector<u8>> mips;
    buildMips(pixels, static_cast<u32>(w), static_cast<u32>(h), usage == TextureUsage::Color, usage == TextureUsage::Normal, mips);
    stbi_image_free(pixels);

    out.width = static_cast<u32>(w);
    out.height = static_cast<u32>(h);
    out.mipLevels = static_cast<u32>(mips.size());
    out.format = usage == TextureUsage::Normal ? rhi::Format::Bc5Unorm
                 : usage == TextureUsage::Color ? rhi::Format::Bc7Srgb
                                                : rhi::Format::Bc7Unorm;
    out.data.clear();
    for (u32 mip = 0; mip < out.mipLevels; ++mip)
        compress(mips[mip], std::max(1u, out.width >> mip), std::max(1u, out.height >> mip), usage, out.data);

    Header hd{kMagic, kVersion, kFamilyBc, static_cast<u32>(out.format), out.width, out.height, out.mipLevels, 0};
    std::string file(reinterpret_cast<const char*>(&hd), sizeof(hd));
    file.append(reinterpret_cast<const char*>(out.data.data()), out.data.size());
    if (!createDirectories(textureCacheFolder()) || !writeTextFile(written, file))
        MEALKIT_WARN("Texture cache: cannot write %s", written.c_str());
    MEALKIT_INFO("Texture cache: compressed %ux%u, %u mips -> %s", out.width, out.height, out.mipLevels, name.c_str());
    return true;
#else
    error = "texture " + name + " is not in the texture cache (export the game again)";
    return false;
#endif
}

} // namespace mealkit::asset
