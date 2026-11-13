#pragma once

#include "mealkit/core/Math.h"
#include "mealkit/asset/MeshData.h"

#include <string>
#include <vector>

// Heightmap terrain (scene kind scene::kTypeTerrain): 16-bit PNG heights filling the object's box.
// Sample (x, z) sits at center + (-size.x/2 + x * size.x/(n-1), -size.y/2 + s/65535 * size.y, -size.z/2 + z * size.z/(n-1)).
// The same samples feed the collision (physics::PhysicsWorld::createHeightField, integer decode, so the same on every PC),
// and the view's chunk meshes (same diagonal as Jolt's triangles). Painting is MkRender's (mkterrain), which writes the PNG.
namespace mealkit::asset {

// Samples in grid order: index = z * size + x, z = 0 at the -z edge (PNG rows are flipped on load/save: the image's top row is +z)
struct Heightmap {
    u32 size = 0; // samples per side
    std::vector<u16> samples;

    bool valid() const { return size >= 2 && samples.size() == static_cast<size_t>(size) * size; }
    u16 at(u32 x, u32 z) const { return samples[static_cast<size_t>(z) * size + x]; }
    u64 hash() const;
};

Heightmap flatHeightmap(u32 size, u16 value);
// PNG (16-bit or 8-bit, any channel count: the first channel is read) → heightmap. Must be square
bool decodeHeightmap(const u8* png, size_t bytes, Heightmap& out, std::string& error);
bool loadHeightmap(const std::string& file, Heightmap& out, std::string& error);

// Heights in meters above the box bottom, grid order (collision)
void terrainHeights(const Heightmap& map, f32 height, std::vector<f32>& out);
// Surface height (world y) at world x/z on the collision triangles. false outside the terrain
bool terrainHeightAt(const Heightmap& map, Vec3 center, Vec3 size, f32 x, f32 z, f32* y);
// First hit of a ray (dir normalized) with the surface: distance, or -1 if none within maxDistance (editor picking)
f32 raycastTerrain(const Heightmap& map, Vec3 center, Vec3 size, Vec3 origin, Vec3 dir, f32 maxDistance);

// ---------------------------------------------------------------- View: chunk meshes
inline constexpr u32 kTerrainChunkCells = 32; // cells per chunk side at LOD 0
inline constexpr u32 kTerrainLods = 3;        // LOD n skips 2^n samples
u32 terrainChunks(const Heightmap& map);      // chunks per side
// One chunk's mesh, vertices relative to the box center (instance at the center, scale 1), with skirts that hide cracks between LODs
void buildTerrainChunk(const Heightmap& map, Vec3 size, u32 cx, u32 cz, u32 lod, MeshData& out);
// LOD by distance from the camera to the chunk center (in chunk widths)
u32 terrainLod(f32 distance, f32 chunkWidth);

} // namespace mealkit::asset
