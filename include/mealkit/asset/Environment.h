#pragma once

#include "mealkit/core/Math.h"

#include <string>
#include <vector>

// Sky HDRI → ambient/reflections (docs/ROADMAP.md stage 2 "Sky HDRI and ambient", docs/DECISIONS.md 2026-10-26 HDRI).
// Baked from the scene environment's single `.hdr` (equirectangular Radiance) on load and saved to a file:
//   <project>/build/cache/environments/<content hash>.mkenv   (outside git, rebaked if deleted)
//   - sky cubemap (skybox, one mip)
//   - reflection cubemap 128² + mips (mip k = roughness k / (mip count − 1), GGX importance samples)
//   - 9 ambient spherical harmonic coefficients (Lambert convolution multiplied in, so shaders get the "ambient color" directly from the normal)
// Cube face order/orientation follows DX12/Vulkan rules (+X, −X, +Y, −Y, +Z, −Z, MealKit left-handed y-up), texels are RGB9E5.
// Baking happens only in dev builds (builds with debug UI): shipping builds only read the cache (Export copies it to cache/environments next to the exe).
// View-only, so unrelated to the simulation and content hash.
namespace mealkit::asset {

struct EnvironmentMap {
    f32 sh[9][3] = {}; // ambient color = Σ sh[i] · Y_i(n) (Y: real spherical harmonics orders 0~2)
    u32 skySize = 0;   // sky cube face size (one mip)
    std::vector<u32> sky;
    u32 reflectSize = 0, reflectMips = 0; // reflection cube (per face starting from mip 0)
    std::vector<u32> reflect;
};

// hdrPath (.hdr file) → read from cache, or if missing (dev build) bake and write to cache. On failure, false and a reason
bool loadEnvironment(const std::string& hdrPath, EnvironmentMap& out, std::string& error);

// .hdr files under assets/textures/ (including subfolders) (paths relative to assets like "textures/hdri/x.hdr", sorted by name). Selection list for the editor [Environment] window and lookdev
std::vector<std::string> listHdris(const std::string& assetsRoot);

// Color grading LUT (.cube, 3D LUT exported by Resolve, Photoshop etc.): sRGB values resampled to render::kLutSize³ cells (red fastest,
// render::Renderer::createLut). Trilinear interpolation if sizes differ, honors DOMAIN_MIN/MAX. 1D LUTs and bad files return false and a reason
bool parseCubeLut(const std::string& text, std::vector<f32>& out, std::string& error);
bool loadCubeLut(const std::string& path, std::vector<f32>& out, std::string& error);
// .cube files under assets/lut/ (environment panel selection, kind folder lut)
std::vector<std::string> listLuts(const std::string& assetsRoot);
// Writes a neutral LUT as .cube (placed with the grading capture as a starting point in external tools)
bool saveNeutralCubeLut(const std::string& path);

// Cache folder: written to <project>/build/cache/environments; when reading, also cache/environments next to the exe (shipping)
std::string environmentCacheFolder();

// Reflection probes (docs/ROADMAP.md stage 2 "Reflection probes"): at a scene reflection_probe object's position, the editor renders the scene to six cube faces
// and reads them back (editor/ProbeBaker), and they're baked here into a reflection cube + GGX mips. Files are assets/probes/<scene>_<object id>.mkprobe (committed:
// shipping reads these files without baking). Same layout as the sky HDRI reflection cube (RGB9E5, per face starting from mip 0)
struct ProbeMap {
    u32 size = 0, mips = 0;
    std::vector<u32> reflect;
};
// faces: six faces (+X −X +Y −Y +Z −Z, cubeDirection orientation) × faceSize² × linear rgb. Size is faceSize, mips down to 4
void bakeProbe(const f32* faces, u32 faceSize, ProbeMap& out);
bool saveProbe(const std::string& path, const ProbeMap& probe);
bool loadProbe(const std::string& path, ProbeMap& out, std::string& error);

// --- Baking parts (unit tests)
u32 packRgb9e5(f32 r, f32 g, f32 b); // negatives become 0, overly large values the max
void unpackRgb9e5(u32 v, f32 out[3]);
Vec3 cubeDirection(u32 face, f32 s, f32 t); // face 0..5, s/t = −1..1 (texel center) → normalized direction
// Bake from an equirectangular image (rgb float, width × height) (no file/cache)
void bakeEnvironment(const f32* rgb, u32 width, u32 height, EnvironmentMap& out);
Vec3 evalSh(const f32 (&sh)[9][3], Vec3 n); // same computation as the shader

} // namespace mealkit::asset
