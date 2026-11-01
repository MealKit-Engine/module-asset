#pragma once

#include "mealkit/asset/Animation.h"
#include "mealkit/core/Reflect.h"
#include "mealkit/sim/Table.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Animation state table (docs/ROADMAP.md stage 4 "Animation state table + animation events"): idle/run/jump/attack switching as data.
// View-side only (like samplePose): the result never enters the simulation. No graph editor; edited in the [Tables] panel.
//
// assets/tables/animation.table, one row per state. Row id = "<set>.<state>" (set = character kind). The first row of a set is its entry state.
//   row pawn.idle
//   clip idle
//   transitions move if speed > 0.5; wave on push
//   row pawn.move
//   clip idle 0 run 3              ← 1D blend: "clip value" pairs, blended by blendParam
//   blendParam speed
//   transitions idle if speed < 0.3
//   events 0 Footstep 0.5 Footstep ← "cycle fraction (0..1) feedback row" pairs
//   row pawn.wave
//   clip wave
//   loop false
//   transitions idle if end
// Transitions are checked in order, the first match wins: "<state> if <cond> [and <cond>...]" (cond: param > < >= <= == != number,
// "param" (> 0.5), "!param" (< 0.5), "end" (non-looping clip finished)) or "<state> on <trigger>". Separated by ";".
// Parameters (speed, grounded, ...) and triggers are set by game view code every frame (AnimParams).
namespace mealkit::asset {

MEALKIT_STRUCT(AnimState) {
    MEALKIT_FIELD(std::string, clip, {}, "Clip, or 1D blend \"clip value clip value\"");
    MEALKIT_FIELD(std::string, blendParam, {}, "Blend parameter (1D blend)");
    MEALKIT_FIELD(bool, loop, true, "Loop");
    MEALKIT_FIELD(f32, speed, 1.0f, "Playback speed");
    MEALKIT_FIELD(f32, blendIn, 0.15f, "Blend time when entering (s)");
    MEALKIT_FIELD(std::string, transitions, {}, "Transitions \"state if speed > 1; state on trigger\"");
    MEALKIT_FIELD(std::string, events, {}, "Events \"cycle fraction feedback row\" pairs");
};
inline constexpr const char* kAnimationTable = "tables/animation.table"; // relative to assets

// Values game view code gives the state table each frame. Triggers are consumed by the next updateAnim
struct AnimParams {
    std::vector<std::pair<std::string, f32>> values;
    std::vector<std::string> triggers;

    void set(std::string_view name, f32 value);
    void set(std::string_view name, bool value) { set(name, value ? 1.0f : 0.0f); }
    void trigger(std::string_view name) { triggers.emplace_back(name); }
    f32 get(std::string_view name) const; // 0 if never set
    bool fired(std::string_view name) const;
};

// Parsed table: sets of states with resolved transition targets
class AnimStates {
public:
    struct Cond {
        enum Op : u8 { Greater, Less, GreaterEq, LessEq, Equal, NotEqual, End };
        Op op = Greater;
        std::string param;
        f32 value = 0.0f;
    };
    struct Transition {
        u32 target = 0;
        std::string trigger; // "on <trigger>" (conds empty)
        std::vector<Cond> conds;
    };
    struct Point {
        std::string clip;
        f32 at = 0.0f;
    };
    struct State {
        std::string name;
        std::vector<Point> points; // 1 = single clip, more = 1D blend (sorted by at)
        std::string blendParam;
        bool loop = true;
        f32 speed = 1.0f;
        f32 blendIn = 0.15f;
        std::vector<Transition> transitions;
        std::vector<std::pair<f32, std::string>> events; // cycle fraction, feedback row
    };
    struct Set {
        std::string name;
        std::vector<State> states;
    };

    // Parses rows (warnings: unknown target states, bad conditions)
    void build(const sim::Table<AnimState>& table, std::string* warnings = nullptr);
    // Reads assets/tables/animation.table (returns false if missing: no sets)
    bool load();
    // Rereads the table when the file changed (checked every 0.25 s of dt). True if reloaded
    bool poll(f64 dt);

    i32 findSet(std::string_view name) const; // -1 if missing
    const Set& set(u32 index) const { return m_sets[index]; }
    u32 setCount() const { return static_cast<u32>(m_sets.size()); }

private:
    std::vector<Set> m_sets;
    std::string m_path;
    unsigned long long m_time = 0;
    f64 m_check = 0.0;
};

// One character's playback state (view side, kept between frames). set = -1 means not started
struct AnimInstance {
    i32 set = -1;
    u32 state = 0, from = 0;
    f32 phase = 0.0f, fromPhase = 0.0f; // cycles played (looping: wraps by 1)
    f32 blend = 0.0f, fromBlend = 0.0f; // blend parameter value at the last update
    f32 fade = 1.0f;                    // 0..1 progress from `from` to `state`
    f32 fadeSeconds = 0.0f;
};

// Advances one character: transitions, playback, events (feedback row names fired this update are appended to events).
// set changes restart at the entry state. Consumes params.triggers
void updateAnim(const AnimStates& states, i32 set, const std::vector<AnimationClip>& clips, AnimInstance& anim,
                AnimParams& params, f32 dt, std::vector<std::string>* events = nullptr);
// Current pose (rest pose if not started)
void animPose(const AnimStates& states, const Skeleton& skeleton, const std::vector<AnimationClip>& clips,
              const AnimInstance& anim, std::vector<JointPose>& out);

} // namespace mealkit::asset
