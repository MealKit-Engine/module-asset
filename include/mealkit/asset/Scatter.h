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

} // namespace mealkit::asset
