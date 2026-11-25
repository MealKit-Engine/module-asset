#include "mealkit/asset/AnimStates.h"

#include "mealkit/asset/AnimIk.h"

#include "mealkit/core/Log.h"
#include "mealkit/data/Tables.h"
#include "mealkit/platform/Paths.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace mealkit::asset {
namespace {

constexpr f64 kTableCheckSeconds = 0.25;

std::vector<std::string_view> words(std::string_view s) {
    std::vector<std::string_view> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
        const size_t start = i;
        while (i < s.size() && s[i] != ' ' && s[i] != '\t') ++i;
        if (i > start) out.push_back(s.substr(start, i - start));
    }
    return out;
}

bool number(std::string_view s, f32& out) {
    // strtof: Apple's libc++ has no float std::from_chars
    const std::string text(s);
    char* end = nullptr;
    out = std::strtof(text.c_str(), &end);
    return !text.empty() && end == text.c_str() + text.size();
}

void warn(std::string* warnings, const std::string& text) {
    if (!warnings) return;
    if (!warnings->empty()) *warnings += "; ";
    *warnings += text;
}

// "speed > 0.5", "grounded", "!grounded", "end"
bool parseCond(const std::vector<std::string_view>& w, AnimStates::Cond& out) {
    using Cond = AnimStates::Cond;
    if (w.size() == 1) {
        if (w[0] == "end") return out.op = Cond::End, true;
        const bool negate = w[0][0] == '!';
        out.param = std::string(w[0].substr(negate ? 1 : 0));
        out.op = negate ? Cond::Less : Cond::Greater;
        out.value = 0.5f;
        return !out.param.empty();
    }
    if (w.size() != 3 || !number(w[2], out.value)) return false;
    static constexpr std::pair<std::string_view, Cond::Op> kOps[] = {{">", Cond::Greater},   {"<", Cond::Less},
                                                                     {">=", Cond::GreaterEq}, {"<=", Cond::LessEq},
                                                                     {"==", Cond::Equal},     {"!=", Cond::NotEqual}};
    for (const auto& [text, op] : kOps)
        if (w[1] == text) {
            out.op = op;
            out.param = std::string(w[0]);
            return true;
        }
    return false;
}

const AnimationClip* clipOf(const std::vector<AnimationClip>& clips, const AnimStates::Point& p) { return findClip(clips, p.clip); }

struct Weight {
    u32 point;
    f32 weight;
};

// Blend weights of a state's points for the parameter values: 1D = the two bracketing points, 2D = inverse squared distance
void weightsOf(const AnimStates::State& s, f32 x, f32 y, std::vector<Weight>& out) {
    out.clear();
    const u32 n = static_cast<u32>(s.points.size());
    if (n == 0) return;
    if (n == 1) {
        out.push_back({0, 1.0f});
        return;
    }
    if (!s.blendParam2.empty()) {
        f32 total = 0.0f;
        for (u32 i = 0; i < n; ++i) {
            const f32 dx = x - s.points[i].at, dy = y - s.points[i].at2;
            const f32 d2 = dx * dx + dy * dy;
            if (d2 < 1e-8f) { // on a point: exactly that clip
                out.assign(1, {i, 1.0f});
                return;
            }
            out.push_back({i, 1.0f / d2});
            total += out.back().weight;
        }
        for (Weight& w : out) w.weight /= total;
        return;
    }
    if (x <= s.points[0].at) {
        out.push_back({0, 1.0f});
        return;
    }
    if (x >= s.points[n - 1].at) {
        out.push_back({n - 1, 1.0f});
        return;
    }
    for (u32 i = 1; i < n; ++i)
        if (x <= s.points[i].at) {
            const f32 span = s.points[i].at - s.points[i - 1].at;
            const f32 w = span > 0.0f ? (x - s.points[i - 1].at) / span : 0.0f;
            out.push_back({i - 1, 1.0f - w});
            if (w > 0.0f) out.push_back({i, w});
            return;
        }
}

// Seconds of one cycle (blend: weighted mean of the active clips' lengths)
f32 cycleSeconds(const AnimStates::State& s, const std::vector<AnimationClip>& clips, f32 x, f32 y) {
    thread_local std::vector<Weight> weights;
    weightsOf(s, x, y, weights);
    f32 seconds = 0.0f;
    for (const Weight& w : weights) {
        const AnimationClip* c = clipOf(clips, s.points[w.point]);
        seconds += (c ? c->duration : 0.0f) * w.weight;
    }
    return seconds;
}

bool finished(const AnimStates::State& s, f32 phase) { return !s.loop && phase >= 1.0f; }

bool passes(const AnimStates::Transition& t, const AnimStates::State& s, f32 phase, const AnimParams& params) {
    using Cond = AnimStates::Cond;
    if (!t.trigger.empty()) return params.fired(t.trigger);
    for (const Cond& c : t.conds) {
        const f32 v = c.op == Cond::End ? 0.0f : params.get(c.param);
        bool ok = false;
        switch (c.op) {
        case Cond::Greater: ok = v > c.value; break;
        case Cond::Less: ok = v < c.value; break;
        case Cond::GreaterEq: ok = v >= c.value; break;
        case Cond::LessEq: ok = v <= c.value; break;
        case Cond::Equal: ok = v == c.value; break;
        case Cond::NotEqual: ok = v != c.value; break;
        case Cond::End: ok = finished(s, phase); break;
        }
        if (!ok) return false;
    }
    return !t.conds.empty();
}

// Advances phase by dt; fires events crossed in (old, new]
void advance(const AnimStates::State& s, const std::vector<AnimationClip>& clips, f32 blend, f32 blend2, f32 dt, f32& phase,
             std::vector<std::string>* events, bool* wrapped = nullptr) {
    const f32 seconds = cycleSeconds(s, clips, blend, blend2);
    const f32 before = phase;
    phase += seconds > 0.0f ? dt * s.speed / seconds : 0.0f;
    if (!s.loop) phase = std::min(phase, 1.0f);
    if (events && phase > before)
        for (const auto& [at, name] : s.events)
            for (f32 k = std::floor(before - at) + 1.0f; k + at <= phase; k += 1.0f)
                if (k + at > before) events->push_back(name);
    if (s.loop && phase >= 1.0f) {
        phase -= std::floor(phase);
        if (wrapped) *wrapped = true;
    }
}

void statePose(const AnimStates::State& s, const Skeleton& skeleton, const std::vector<AnimationClip>& clips, f32 phase,
               f32 blend, f32 blend2, std::vector<JointPose>& out) {
    thread_local std::vector<JointPose> next;
    thread_local std::vector<Weight> weights;
    if (s.points.empty()) return restPose(skeleton, out);
    weightsOf(s, blend, blend2, weights);
    f32 sum = 0.0f;
    for (const Weight& w : weights) {
        if (w.weight <= 0.0f) continue;
        std::vector<JointPose>& dst = sum == 0.0f ? out : next;
        const AnimationClip* clip = clipOf(clips, s.points[w.point]);
        if (clip) samplePose(skeleton, *clip, phase * clip->duration, s.loop, dst);
        else restPose(skeleton, dst);
        if (sum > 0.0f) blendPoses(out, next, w.weight / (sum + w.weight), out); // per joint, so out may alias a
        sum += w.weight;
    }
    if (sum == 0.0f) restPose(skeleton, out);
    if (!s.rootMotion.empty()) stripRootMotion(skeleton, findJoint(skeleton, s.rootMotion), out);
}

} // namespace

void AnimParams::set(std::string_view name, f32 value) {
    for (auto& [n, v] : values)
        if (n == name) {
            v = value;
            return;
        }
    values.emplace_back(name, value);
}

f32 AnimParams::get(std::string_view name) const {
    for (const auto& [n, v] : values)
        if (n == name) return v;
    return 0.0f;
}

bool AnimParams::fired(std::string_view name) const {
    return std::find(triggers.begin(), triggers.end(), name) != triggers.end();
}

void AnimStates::build(const sim::Table<AnimState>& table, std::string* warnings) {
    m_sets.clear();
    std::vector<std::pair<u32, u32>> where(table.size()); // row → (set, state)
    for (u32 r = 0; r < table.size(); ++r) {
        const std::string& id = table.id(r);
        const size_t dot = id.find('.');
        if (dot == std::string::npos || dot == 0 || dot + 1 == id.size()) {
            warn(warnings, "row " + id + ": id must be <set>.<state>");
            where[r] = {~0u, 0};
            continue;
        }
        const std::string setName = id.substr(0, dot);
        i32 s = findSet(setName);
        if (s < 0) {
            s = static_cast<i32>(m_sets.size());
            m_sets.push_back({setName, {}});
        }
        Set& set = m_sets[static_cast<u32>(s)];
        const AnimState& row = table[r];
        State state;
        state.name = id.substr(dot + 1);
        state.blendParam = row.blendParam;
        state.blendParam2 = row.blendParam2;
        state.rootMotion = row.rootMotion;
        state.loop = row.loop;
        state.speed = row.speed;
        state.blendIn = std::max(0.0f, row.blendIn);
        const std::vector<std::string_view> clip = words(row.clip);
        const bool blend2d = !row.blendParam2.empty();
        const size_t stride = blend2d ? 3 : 2;
        if (clip.size() == 1) {
            state.points.push_back({std::string(clip[0]), 0.0f, 0.0f});
        } else {
            for (size_t i = 0; i + stride - 1 < clip.size(); i += stride) {
                Point p{std::string(clip[i]), 0.0f, 0.0f};
                if (number(clip[i + 1], p.at) && (!blend2d || number(clip[i + 2], p.at2))) state.points.push_back(std::move(p));
                else warn(warnings, "row " + id + ": blend value for '" + std::string(clip[i]) + "' is not a number");
            }
            if (clip.size() % stride) warn(warnings, "row " + id + ": blend clip without " + (blend2d ? "x and y values" : "a value"));
            if (!blend2d)
                std::stable_sort(state.points.begin(), state.points.end(), [](const Point& x, const Point& y) { return x.at < y.at; });
        }
        const std::vector<std::string_view> ev = words(row.events);
        for (size_t i = 0; i + 1 < ev.size(); i += 2) {
            f32 at = 0.0f;
            if (number(ev[i], at) && at >= 0.0f && at < 1.0f) state.events.emplace_back(at, std::string(ev[i + 1]));
            else warn(warnings, "row " + id + ": event time '" + std::string(ev[i]) + "' must be 0 to below 1");
        }
        where[r] = {static_cast<u32>(s), static_cast<u32>(set.states.size())};
        set.states.push_back(std::move(state));
    }
    // Transitions after all states exist (targets can be later rows)
    for (u32 r = 0; r < table.size(); ++r) {
        if (where[r].first == ~0u) continue;
        Set& set = m_sets[where[r].first];
        State& state = set.states[where[r].second];
        std::string_view rest = table[r].transitions;
        while (!rest.empty()) {
            const size_t end = rest.find(';');
            std::string_view item = rest.substr(0, end);
            while (!item.empty() && (item.front() == ' ' || item.front() == '\t')) item.remove_prefix(1);
            while (!item.empty() && (item.back() == ' ' || item.back() == '\t')) item.remove_suffix(1);
            rest = end == std::string_view::npos ? std::string_view{} : rest.substr(end + 1);
            const std::vector<std::string_view> w = words(item);
            if (w.empty()) continue;
            const std::string bad = "row " + table.id(r) + ": transition '" + std::string(item) + "'";
            Transition t;
            const auto target = std::find_if(set.states.begin(), set.states.end(), [&](const State& s) { return s.name == w[0]; });
            if (target == set.states.end()) {
                warn(warnings, bad + " goes to unknown state");
                continue;
            }
            t.target = static_cast<u32>(target - set.states.begin());
            bool ok = w.size() >= 3 && (w[1] == "if" || w[1] == "on");
            if (ok && w[1] == "on") {
                ok = w.size() == 3;
                t.trigger = std::string(w[2]);
            } else if (ok) {
                std::vector<std::string_view> part;
                for (size_t i = 2; i <= w.size() && ok; ++i) {
                    if (i == w.size() || w[i] == "and") {
                        Cond c;
                        ok = parseCond(part, c);
                        t.conds.push_back(std::move(c));
                        part.clear();
                    } else {
                        part.push_back(w[i]);
                    }
                }
            }
            if (ok) state.transitions.push_back(std::move(t));
            else warn(warnings, bad + " is not \"<state> if <cond>\" or \"<state> on <trigger>\"");
        }
    }
}

bool AnimStates::load() {
    m_path = findAsset(std::string("assets/") + kAnimationTable);
    m_time = m_path.empty() ? 0 : fileWriteTime(m_path);
    sim::Table<AnimState> table;
    if (!m_path.empty()) data::loadTable(table, kAnimationTable);
    std::string warnings;
    build(table, &warnings);
    if (!warnings.empty()) MEALKIT_WARN("assets/%s: %s", kAnimationTable, warnings.c_str());
    return !m_path.empty();
}

bool AnimStates::poll(f64 dt) {
    m_check -= dt;
    if (m_check > 0.0) return false;
    m_check = kTableCheckSeconds;
    const std::string path = findAsset(std::string("assets/") + kAnimationTable);
    const unsigned long long t = path.empty() ? 0 : fileWriteTime(path);
    if (path == m_path && t == m_time) return false;
    load();
    MEALKIT_INFO("Animation: %u set(s) from %s", setCount(), path.empty() ? "(no animation.table)" : path.c_str());
    return true;
}

i32 AnimStates::findSet(std::string_view name) const {
    for (size_t i = 0; i < m_sets.size(); ++i)
        if (m_sets[i].name == name) return static_cast<i32>(i);
    return -1;
}

void updateAnim(const AnimStates& states, i32 set, const std::vector<AnimationClip>& clips, AnimInstance& anim,
                AnimParams& params, f32 dt, std::vector<std::string>* events) {
    if (set < 0 || static_cast<u32>(set) >= states.setCount() || states.set(static_cast<u32>(set)).states.empty()) {
        anim = {};
        params.triggers.clear();
        return;
    }
    const AnimStates::Set& s = states.set(static_cast<u32>(set));
    if (anim.set != set || anim.state >= s.states.size() || anim.from >= s.states.size()) anim = {set};
    const AnimStates::State* current = &s.states[anim.state];
    for (const AnimStates::Transition& t : current->transitions) {
        if (t.target == anim.state || !passes(t, *current, anim.phase, params)) continue;
        anim.from = anim.state;
        anim.fromPhase = anim.phase;
        anim.fromBlend = anim.blend;
        anim.fromBlend2 = anim.blend2;
        anim.state = t.target;
        anim.phase = 0.0f;
        current = &s.states[anim.state];
        anim.fadeSeconds = current->blendIn;
        anim.fade = anim.fadeSeconds > 0.0f ? 0.0f : 1.0f;
        break;
    }
    params.triggers.clear();
    anim.blend = current->blendParam.empty() ? 0.0f : params.get(current->blendParam);
    anim.blend2 = current->blendParam2.empty() ? 0.0f : params.get(current->blendParam2);
    anim.prevPhase = anim.phase;
    anim.wrapped = false;
    advance(*current, clips, anim.blend, anim.blend2, dt, anim.phase, events, &anim.wrapped);
    if (anim.fade < 1.0f) {
        advance(s.states[anim.from], clips, anim.fromBlend, anim.fromBlend2, dt, anim.fromPhase, nullptr); // fading-out state fires no events
        anim.fade = std::min(1.0f, anim.fade + dt / anim.fadeSeconds);
    }
}

void animPose(const AnimStates& states, const Skeleton& skeleton, const std::vector<AnimationClip>& clips,
              const AnimInstance& anim, std::vector<JointPose>& out) {
    if (anim.set < 0 || static_cast<u32>(anim.set) >= states.setCount()) return restPose(skeleton, out);
    const AnimStates::Set& s = states.set(static_cast<u32>(anim.set));
    if (anim.state >= s.states.size()) return restPose(skeleton, out);
    statePose(s.states[anim.state], skeleton, clips, anim.phase, anim.blend, anim.blend2, out);
    if (anim.fade >= 1.0f || anim.from >= s.states.size()) return;
    thread_local std::vector<JointPose> before;
    statePose(s.states[anim.from], skeleton, clips, anim.fromPhase, anim.fromBlend, anim.fromBlend2, before);
    blendPoses(before, out, anim.fade, out);
}

Vec3 animRootDelta(const AnimStates& states, const Skeleton& skeleton, const std::vector<AnimationClip>& clips, const AnimInstance& anim) {
    if (anim.set < 0 || static_cast<u32>(anim.set) >= states.setCount()) return {};
    const AnimStates::Set& set = states.set(static_cast<u32>(anim.set));
    if (anim.state >= set.states.size()) return {};
    const AnimStates::State& s = set.states[anim.state];
    if (s.rootMotion.empty()) return {};
    const i32 joint = findJoint(skeleton, s.rootMotion);
    if (joint < 0) return {};
    thread_local std::vector<Weight> weights;
    weightsOf(s, anim.blend, anim.blend2, weights);
    Vec3 delta{};
    for (const Weight& w : weights) {
        const AnimationClip* clip = clipOf(clips, s.points[w.point]);
        if (!clip) continue;
        const f32 to = (anim.phase + (anim.wrapped ? 1.0f : 0.0f)) * clip->duration;
        Vec3 d = clipRootDelta(skeleton, *clip, joint, anim.prevPhase * clip->duration, to, s.loop);
        d.y = 0.0f; // only horizontal travel is taken out of the pose
        delta += d * w.weight;
    }
    return delta;
}

} // namespace mealkit::asset
