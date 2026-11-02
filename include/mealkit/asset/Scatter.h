#pragma once

#include "mealkit/core/Math.h"

#include <cmath>
#include <string>
#include <vector>

// Scatter layers (scene kind scene::kTypeScatter): thousands of view-only copies of one model (grass, rocks, small props) painted onto
// terrain or floors in the editor. Not collision, not simulation state, not in the content hash.
// Instances live in a text file (assets/terrain/<scene>_<id>.scatter, "@format scatter 1", one "i x y z yaw scale" line each),
// grouped into square cells for drawing so far cells are skipped as a whole.
namespace mealkit::asset {

struct ScatterInstance {
    Vec3 position; // world
    f32 yaw = 0.0f;   // radians around y
    f32 scale = 1.0f; // uniform
};

inline constexpr f32 kScatterCellSize = 32.0f; // m

struct ScatterCell {
    Vec3 center; // xz center of the cell, y = mean height of its instances
    u32 first = 0, count = 0; // range in Scatter::order
};

struct Scatter {
    std::vector<ScatterInstance> items;
    // Draw grouping (rebuilt by groupScatter when items change)
    std::vector<u32> order; // item indices sorted by cell
    std::vector<ScatterCell> cells;
    bool grouped = false;
};

bool loadScatter(const std::string& file, Scatter& out, std::string& error);
bool saveScatter(const std::string& file, const Scatter& scatter, std::string& error);
void groupScatter(Scatter& scatter);

// Editor painting. Adds up to maxAdd instances at random points in the circle (x/z) where floorAt finds ground,
// keeping spacing 1/sqrt(density) * 0.75 to other instances. Returns the number added
struct ScatterPaint {
    Vec3 center;
    f32 radius = 4.0f;
    f32 density = 0.5f; // instances per m²
    f32 minScale = 0.8f, maxScale = 1.2f;
    bool randomYaw = true;
};
// floorAt(x, z, &y): the surface height under a point (false = nothing there)
template <class FloorAt, class Random>
u32 paintScatter(Scatter& scatter, const ScatterPaint& p, u32 maxAdd, FloorAt&& floorAt, Random&& random01);
// Removes instances inside the circle (x/z). Returns the number removed
u32 eraseScatter(Scatter& scatter, Vec3 center, f32 radius);

template <class FloorAt, class Random>
u32 paintScatter(Scatter& scatter, const ScatterPaint& p, u32 maxAdd, FloorAt&& floorAt, Random&& random01) {
    if (p.density <= 0.0f || p.radius <= 0.0f) return 0;
    const f32 spacing = 0.75f / std::sqrt(p.density), spacing2 = spacing * spacing;
    const f32 reach = (p.radius + spacing) * (p.radius + spacing);
    std::vector<u32> near; // instances that can be within spacing of a new point
    for (u32 i = 0; i < scatter.items.size(); ++i) {
        const f32 dx = scatter.items[i].position.x - p.center.x, dz = scatter.items[i].position.z - p.center.z;
        if (dx * dx + dz * dz <= reach) near.push_back(i);
    }
    u32 added = 0;
    for (u32 attempt = 0; attempt < maxAdd * 4 && added < maxAdd; ++attempt) {
        const f32 a = random01() * 6.2831853f, r = p.radius * std::sqrt(random01());
        const f32 x = p.center.x + std::cos(a) * r, z = p.center.z + std::sin(a) * r;
        bool crowded = false;
        for (u32 i : near) {
            const f32 dx = scatter.items[i].position.x - x, dz = scatter.items[i].position.z - z;
            if (dx * dx + dz * dz < spacing2) {
                crowded = true;
                break;
            }
        }
        f32 y = 0.0f;
        if (crowded || !floorAt(x, z, &y)) continue;
        ScatterInstance s;
        s.position = {x, y, z};
        s.yaw = p.randomYaw ? random01() * 6.2831853f : 0.0f;
        s.scale = p.minScale + (p.maxScale - p.minScale) * random01();
        near.push_back(static_cast<u32>(scatter.items.size()));
        scatter.items.push_back(s);
        ++added;
    }
    if (added) scatter.grouped = false;
    return added;
}

} // namespace mealkit::asset
