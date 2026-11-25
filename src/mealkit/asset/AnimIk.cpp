#include "mealkit/asset/AnimIk.h"

#include "mealkit/core/Log.h"
#include "mealkit/data/Tables.h"
#include "mealkit/platform/Paths.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace mealkit::asset {
namespace {

constexpr f32 kEps = 1e-6f;

Quat conjugate(Quat q) { return {-q.x, -q.y, -q.z, q.w}; }

// Shortest rotation taking unit vector a to unit vector b
Quat arc(Vec3 a, Vec3 b) {
    const f32 d = dot(a, b);
    if (d < -0.99999f) { // opposite: turn half a circle around any perpendicular axis
        Vec3 axis = cross(Vec3{1, 0, 0}, a);
        if (lengthSq(axis) < 1e-6f) axis = cross(Vec3{0, 1, 0}, a);
        axis = normalize(axis);
        return {axis.x, axis.y, axis.z, 0.0f};
    }
    const Vec3 c = cross(a, b);
    return normalize(Quat{c.x, c.y, c.z, 1.0f + d});
}

// Rotation/translation forward kinematics in skeleton space (parents come first)
void forward(const Skeleton& skeleton, const std::vector<JointPose>& pose, std::vector<Quat>& rot, std::vector<Vec3>& pos) {
    const size_t n = skeleton.joints.size();
    rot.resize(n);
    pos.resize(n);
    for (size_t j = 0; j < n; ++j) {
        const JointPose& p = j < pose.size() ? pose[j] : skeleton.joints[j].rest;
        const i32 parent = skeleton.joints[j].parent;
        if (parent < 0) {
            rot[j] = p.rotation;
            pos[j] = p.translation;
        } else {
            const size_t k = static_cast<size_t>(parent);
            rot[j] = rot[k] * p.rotation;
            pos[j] = pos[k] + rotate(rot[k], p.translation);
        }
    }
}

bool numbers3(std::string_view text, Vec3& out) {
    std::istringstream in{std::string(text)};
    f32 x, y, z;
    if (!(in >> x >> y >> z)) return false;
    out = {x, y, z};
    return true;
}

// Root joint translation of a clip at (unwrapped) seconds. Looping clips continue their travel every cycle
Vec3 translationAt(const AnimationClip& clip, u32 jointCount, u32 joint, f32 seconds) {
    if (clip.frames == 0 || clip.poses.size() < static_cast<size_t>(clip.frames) * jointCount) return {};
    const f32 frame = std::clamp(seconds * static_cast<f32>(kClipFrameRate), 0.0f, static_cast<f32>(clip.frames - 1));
    const u32 f0 = static_cast<u32>(frame);
    const u32 f1 = std::min(f0 + 1, clip.frames - 1);
    return lerp(clip.frame(f0, jointCount)[joint].translation, clip.frame(f1, jointCount)[joint].translation, frame - static_cast<f32>(f0));
}

Vec3 unwrappedTranslation(const AnimationClip& clip, u32 jointCount, u32 joint, f32 seconds, bool loop) {
    if (!loop || clip.duration <= 0.0f) return translationAt(clip, jointCount, joint, seconds);
    const f32 cycles = std::floor(seconds / clip.duration);
    const f32 local = seconds - cycles * clip.duration;
    const Vec3 travel = translationAt(clip, jointCount, joint, clip.duration) - translationAt(clip, jointCount, joint, 0.0f);
    return translationAt(clip, jointCount, joint, local) + travel * cycles;
}

} // namespace

// IK table ----------------------------------------------------------------------------------------------------------------------------------

void IkChains::build(const sim::Table<IkChain>& table, std::string* warnings) {
    m_entries.clear();
    const auto warn = [&](const std::string& text) {
        if (!warnings) return;
        if (!warnings->empty()) *warnings += "; ";
        *warnings += text;
    };
    for (u32 r = 0; r < table.size(); ++r) {
        const std::string& id = table.id(r);
        const size_t dot = id.find('.');
        if (dot == std::string::npos || dot == 0 || dot + 1 == id.size()) {
            warn("row " + id + ": id must be <set>.<chain>");
            continue;
        }
        Entry e;
        e.set = id.substr(0, dot);
        e.name = id.substr(dot + 1);
        e.row = table[r];
        if (!e.row.pole.empty()) {
            e.hasPole = numbers3(e.row.pole, e.pole) && lengthSq(e.pole) > kEps;
            if (!e.hasPole) warn("row " + id + ": pole '" + e.row.pole + "' is not \"x y z\"");
            else e.pole = normalize(e.pole);
        }
        if (e.row.root.empty() || e.row.mid.empty() || e.row.tip.empty()) {
            warn("row " + id + ": root, mid and tip joint names are needed");
            continue;
        }
        m_entries.push_back(std::move(e));
    }
}

bool IkChains::load() {
    const std::string path = findAsset(std::string("assets/") + kIkTable);
    sim::Table<IkChain> table;
    if (!path.empty()) data::loadTable(table, kIkTable);
    std::string warnings;
    build(table, &warnings);
    if (!warnings.empty()) MEALKIT_WARN("assets/%s: %s", kIkTable, warnings.c_str());
    return !path.empty();
}

i32 findJoint(const Skeleton& skeleton, std::string_view name) {
    for (size_t j = 0; j < skeleton.joints.size(); ++j)
        if (skeleton.joints[j].name == name) return static_cast<i32>(j);
    return -1;
}

IkJoints IkChains::find(std::string_view set, std::string_view chain, const Skeleton& skeleton) const {
    for (const Entry& e : m_entries) {
        if (e.set != set || e.name != chain) continue;
        IkJoints out;
        out.root = findJoint(skeleton, e.row.root);
        out.mid = findJoint(skeleton, e.row.mid);
        out.tip = findJoint(skeleton, e.row.tip);
        out.pole = e.pole;
        out.hasPole = e.hasPole;
        out.weight = std::clamp(e.row.weight, 0.0f, 1.0f);
        out.ankle = e.row.ankle;
        return out;
    }
    return {};
}

std::vector<std::string> IkChains::names(std::string_view set) const {
    std::vector<std::string> out;
    for (const Entry& e : m_entries)
        if (e.set == set) out.push_back(e.name);
    return out;
}

// Solver ------------------------------------------------------------------------------------------------------------------------------------

Vec3 jointPosition(const Skeleton& skeleton, const std::vector<JointPose>& pose, i32 joint) {
    if (joint < 0 || static_cast<size_t>(joint) >= skeleton.joints.size()) return {};
    std::vector<Quat> rot;
    std::vector<Vec3> pos;
    forward(skeleton, pose, rot, pos);
    return pos[static_cast<size_t>(joint)];
}

bool solveTwoBone(const Skeleton& skeleton, const IkJoints& chain, Vec3 target, f32 weight, std::vector<JointPose>& pose) {
    if (!chain.valid() || weight <= 0.0f || pose.size() != skeleton.joints.size()) return false;
    const size_t ri = static_cast<size_t>(chain.root), mi = static_cast<size_t>(chain.mid), ti = static_cast<size_t>(chain.tip);
    thread_local std::vector<Quat> rot;
    thread_local std::vector<Vec3> pos;
    forward(skeleton, pose, rot, pos);
    const Vec3 a = pos[ri], b = pos[mi], c = pos[ti];
    const f32 l1 = length(b - a), l2 = length(c - b);
    const Vec3 toTarget = target - a;
    const f32 dist = length(toTarget);
    if (l1 < kEps || l2 < kEps || dist < kEps) return false;

    const Vec3 u = toTarget / dist;
    const f32 reachMax = (l1 + l2) * 0.9999f, reachMin = std::fabs(l1 - l2) + 1e-4f;
    const f32 d = std::clamp(dist, reachMin, reachMax);
    // Bend direction: pole hint or the animated bend, made perpendicular to the reach axis
    Vec3 bend = chain.hasPole ? chain.pole : (b - a);
    Vec3 perp = bend - u * dot(bend, u);
    if (lengthSq(perp) < 1e-8f) perp = (b - a) - u * dot(b - a, u);
    if (lengthSq(perp) < 1e-8f) perp = cross(u, std::fabs(u.y) < 0.9f ? Vec3{0, 1, 0} : Vec3{1, 0, 0});
    perp = normalize(perp);
    const f32 x = (l1 * l1 - l2 * l2 + d * d) / (2.0f * d);
    const f32 h = std::sqrt(std::max(0.0f, l1 * l1 - x * x));
    const Vec3 midGoal = a + u * x + perp * h;
    const Vec3 tipGoal = a + u * d;

    const Quat q1 = arc(normalize(b - a), normalize(midGoal - a));
    const Vec3 tipAfterRoot = a + rotate(q1, c - a);
    const Quat q2 = arc(normalize(tipAfterRoot - midGoal), normalize(tipGoal - midGoal));

    const i32 rootParent = skeleton.joints[ri].parent;
    const Quat parentRot = rootParent >= 0 ? rot[static_cast<size_t>(rootParent)] : Quat{};
    const Quat rootWorld = q1 * rot[ri];
    const Quat midWorld = q2 * q1 * rot[mi];
    const Quat rootLocal = conjugate(parentRot) * rootWorld;
    const Quat midLocal = conjugate(rootWorld) * midWorld;
    weight = std::min(weight, 1.0f);
    pose[ri].rotation = nlerp(pose[ri].rotation, rootLocal, weight);
    pose[mi].rotation = nlerp(pose[mi].rotation, midLocal, weight);
    return dist <= l1 + l2;
}

void offsetJoint(const Skeleton& skeleton, i32 joint, f32 dy, std::vector<JointPose>& pose) {
    if (joint < 0 || static_cast<size_t>(joint) >= skeleton.joints.size() || pose.size() != skeleton.joints.size()) return;
    thread_local std::vector<Quat> rot;
    thread_local std::vector<Vec3> pos;
    forward(skeleton, pose, rot, pos);
    const i32 parent = skeleton.joints[static_cast<size_t>(joint)].parent;
    const Quat parentRot = parent >= 0 ? rot[static_cast<size_t>(parent)] : Quat{};
    pose[static_cast<size_t>(joint)].translation += rotate(conjugate(parentRot), Vec3{0, dy, 0});
}

Vec3 groundTarget(const Skeleton& skeleton, const IkJoints& chain, const std::vector<JointPose>& pose, f32 groundY) {
    const Vec3 p = jointPosition(skeleton, pose, chain.tip);
    return {p.x, groundY + chain.ankle, p.z};
}

// Root motion -------------------------------------------------------------------------------------------------------------------------------

Vec3 clipRootDelta(const Skeleton& skeleton, const AnimationClip& clip, i32 rootJoint, f32 fromSeconds, f32 toSeconds, bool loop) {
    if (rootJoint < 0 || static_cast<size_t>(rootJoint) >= skeleton.joints.size()) return {};
    const u32 n = static_cast<u32>(skeleton.joints.size()), j = static_cast<u32>(rootJoint);
    return unwrappedTranslation(clip, n, j, toSeconds, loop) - unwrappedTranslation(clip, n, j, fromSeconds, loop);
}

Vec3 clipRootTravel(const Skeleton& skeleton, const AnimationClip& clip, i32 rootJoint) {
    return clipRootDelta(skeleton, clip, rootJoint, 0.0f, clip.duration, false);
}

Vec3 stripRootMotion(const Skeleton& skeleton, i32 rootJoint, std::vector<JointPose>& pose) {
    if (rootJoint < 0 || static_cast<size_t>(rootJoint) >= skeleton.joints.size() || pose.size() != skeleton.joints.size()) return {};
    const Vec3 rest = skeleton.joints[static_cast<size_t>(rootJoint)].rest.translation;
    Vec3& t = pose[static_cast<size_t>(rootJoint)].translation;
    const Vec3 removed{t.x - rest.x, 0.0f, t.z - rest.z};
    t.x = rest.x;
    t.z = rest.z;
    return removed;
}

void RootOffset::update(f32 dt, f32 returnSpeed) {
    const f32 len = length(offset);
    if (len <= 0.0f) return;
    const f32 step = std::max(returnSpeed * dt, len * std::min(1.0f, 4.0f * dt));
    offset = step >= len ? Vec3{} : offset * ((len - step) / len);
}

MotionWarp MotionWarp::make(Vec3 travel, Vec3 target, f32 maxScale) {
    MotionWarp w;
    const f32 lt = std::sqrt(travel.x * travel.x + travel.z * travel.z);
    const f32 lg = std::sqrt(target.x * target.x + target.z * target.z);
    if (lt < 1e-4f) return w; // the clip doesn't travel: nothing to scale
    w.scale = std::clamp(lg / lt, 0.0f, std::max(maxScale, 0.0f));
    if (lg > 1e-4f) w.yaw = std::atan2(-(travel.x * target.z - travel.z * target.x), travel.x * target.x + travel.z * target.z);
    w.active = true;
    return w;
}

Vec3 MotionWarp::apply(Vec3 delta) const {
    if (!active) return delta;
    const f32 c = std::cos(yaw), s = std::sin(yaw);
    return {(delta.x * c + delta.z * s) * scale, delta.y, (-delta.x * s + delta.z * c) * scale};
}

} // namespace mealkit::asset
