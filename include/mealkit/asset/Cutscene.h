#pragma once

#include "mealkit/core/Math.h"
#include "mealkit/core/Reflect.h"
#include "mealkit/sim/Table.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Cutscenes (docs/ROADMAP.md stage 4 "Virtual cameras + cutscenes"): camera animations made in Blender and exported as glTF, only played back.
// View-side only. No timeline editor: one row of assets/tables/cutscenes.table per cutscene, edited in the [Tables] panel.
//   row intro
//   file models/cutscenes/intro.glb   ← glTF with an animated camera node
//   clip                              ← glTF animation name (empty = the first)
//   events 0 CutsceneStart 2.5 Whoosh ← "seconds feedback row" pairs
// The camera node is the first node with a glTF camera. Its parents are taken as static (their rest transform).
namespace mealkit::asset {

MEALKIT_STRUCT(CutsceneRow) {
    MEALKIT_FIELD(std::string, file, {}, "glTF file with an animated camera (relative to assets)");
    MEALKIT_FIELD(std::string, clip, {}, "Animation name (empty: the first)");
    MEALKIT_FIELD(bool, blockInput, true, "Block player input while playing");
    MEALKIT_FIELD(f32, blendIn, 0.0f, "Blend from the game camera (s)");
    MEALKIT_FIELD(f32, blendOut, 0.8f, "Blend back to the game camera (s)");
    MEALKIT_FIELD(std::string, events, {}, "Events \"seconds feedback row\" pairs");
};
inline constexpr const char* kCutsceneTable = "tables/cutscenes.table"; // relative to assets

// One baked camera frame in MealKit coordinates (left-handed, y up). The camera looks along rotate(rotation, +z)
struct CameraKey {
    Vec3 position;
    Quat rotation;
    f32 fovDegrees = 60.0f; // vertical
};

// Camera animation baked at kClipFrameRate (30 Hz) like skeletal clips
struct CameraClip {
    std::string name;
    std::vector<CameraKey> keys; // at least 1
    f32 duration = 0.0f;         // seconds = (keys - 1) / kClipFrameRate
};

// Bakes every animation of the first camera node in a glTF file (file = absolute path). Without animations, one still clip.
bool loadCameraClips(const std::string& file, std::vector<CameraClip>& out, std::string& error);
// Camera at a time in seconds (clamped to the clip)
CameraKey sampleCamera(const CameraClip& clip, f32 seconds);

// Parsed cutscene table with clips loaded on first use
class Cutscenes {
public:
    struct Cutscene {
        std::string name;
        CutsceneRow row;
        std::vector<std::pair<f32, std::string>> events; // seconds, feedback row (sorted)
        std::shared_ptr<const CameraClip> clip;           // null if the file failed to load
    };

    // Parses rows and loads their files (warnings: missing files, bad events, unknown clip names)
    void build(const sim::Table<CutsceneRow>& table, std::string* warnings = nullptr);
    // Reads assets/tables/cutscenes.table (false if missing: no cutscenes)
    bool load();
    // Rereads the table when it changed (checked every 0.25 s of dt). True if reloaded
    bool poll(f64 dt);

    const Cutscene* find(std::string_view name) const; // nullptr if missing
    u32 count() const { return static_cast<u32>(m_cutscenes.size()); }

private:
    std::vector<Cutscene> m_cutscenes;
    std::string m_path;
    unsigned long long m_time = 0;
    f64 m_check = 0.0;
};

} // namespace mealkit::asset
