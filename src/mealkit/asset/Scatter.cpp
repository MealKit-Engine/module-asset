#include "mealkit/asset/Scatter.h"

#include "mealkit/core/TextFormat.h"
#include "mealkit/platform/Paths.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace mealkit::asset {

bool loadScatter(const std::string& file, Scatter& out, std::string& error) {
    FILE* f = openFile(file, "rb");
    if (!f) {
        error = "cannot open " + file;
        return false;
    }
    out = {};
    char line[256];
    bool format = false;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (!format) {
            u32 version = 0;
            std::string_view text(line);
            while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.remove_suffix(1);
            if (!text::readFormatLine(text, text::kScatter, file, version)) {
                fclose(f);
                error = "not a scatter file";
                return false;
            }
            format = true;
            continue;
        }
        ScatterInstance s;
        if (std::sscanf(line, "i %f %f %f %f %f", &s.position.x, &s.position.y, &s.position.z, &s.yaw, &s.scale) == 5)
            out.items.push_back(s);
    }
    fclose(f);
    return true;
}

bool saveScatter(const std::string& file, const Scatter& scatter, std::string& error) {
    FILE* f = openFile(file, "wb");
    if (!f) {
        error = "cannot write " + file;
        return false;
    }
    fprintf(f, "# MealKit scatter layer: i x y z yaw scale (painted in the editor, view only)\n%s\n",
            text::formatLine(text::kScatter).c_str());
    for (const ScatterInstance& s : scatter.items)
        fprintf(f, "i %.3f %.3f %.3f %.3f %.3f\n", s.position.x, s.position.y, s.position.z, s.yaw, s.scale);
    const bool ok = fflush(f) == 0;
    fclose(f);
    if (!ok) error = "cannot write " + file;
    return ok;
}

void groupScatter(Scatter& scatter) {
    auto key = [](const ScatterInstance& s) {
        const i64 cx = static_cast<i64>(std::floor(s.position.x / kScatterCellSize));
        const i64 cz = static_cast<i64>(std::floor(s.position.z / kScatterCellSize));
        return (cz << 32) ^ (cx & 0xFFFFFFFF);
    };
    scatter.order.resize(scatter.items.size());
    for (u32 i = 0; i < scatter.order.size(); ++i) scatter.order[i] = i;
    std::sort(scatter.order.begin(), scatter.order.end(),
              [&](u32 a, u32 b) { return key(scatter.items[a]) < key(scatter.items[b]); });
    scatter.cells.clear();
    for (u32 i = 0; i < scatter.order.size();) {
        const i64 k = key(scatter.items[scatter.order[i]]);
        ScatterCell cell;
        cell.first = i;
        f32 y = 0.0f;
        while (i < scatter.order.size() && key(scatter.items[scatter.order[i]]) == k) y += scatter.items[scatter.order[i++]].position.y;
        cell.count = i - cell.first;
        const ScatterInstance& s = scatter.items[scatter.order[cell.first]];
        cell.center = {(std::floor(s.position.x / kScatterCellSize) + 0.5f) * kScatterCellSize, y / static_cast<f32>(cell.count),
                       (std::floor(s.position.z / kScatterCellSize) + 0.5f) * kScatterCellSize};
        scatter.cells.push_back(cell);
    }
    scatter.grouped = true;
}

} // namespace mealkit::asset
