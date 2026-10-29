#include "mealkit/asset/Animation.h"

#include <algorithm>
#include <cmath>

namespace mealkit::asset {
namespace {

JointPose lerpPose(const JointPose& a, const JointPose& b, f32 t) {
    return {lerp(a.translation, b.translation, t), nlerp(a.rotation, b.rotation, t), lerp(a.scale, b.scale, t)};
}

void framesBetween(const Skeleton& skeleton, const AnimationClip& clip, u32 f0, u32 f1, f32 t, std::vector<JointPose>& out) {
    const u32 count = static_cast<u32>(skeleton.joints.size());
    out.resize(count);
    if (clip.frames == 0 || clip.poses.size() < static_cast<size_t>(clip.frames) * count) {
        restPose(skeleton, out);
        return;
    }
    const JointPose* a = clip.frame(f0, count);
    const JointPose* b = clip.frame(f1, count);
    for (u32 j = 0; j < count; ++j) out[j] = t == 0.0f ? a[j] : lerpPose(a[j], b[j], t);
}

} // namespace

Mat4 composeTrs(const JointPose& p) {
    const Quat q = p.rotation;
    const f32 xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z, xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const f32 wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    Mat4 m; // rows = rotated axes × scale (row vector convention: v * M)
    m.m[0][0] = (1 - 2 * (yy + zz)) * p.scale.x;
    m.m[0][1] = (2 * (xy + wz)) * p.scale.x;
    m.m[0][2] = (2 * (xz - wy)) * p.scale.x;
    m.m[1][0] = (2 * (xy - wz)) * p.scale.y;
    m.m[1][1] = (1 - 2 * (xx + zz)) * p.scale.y;
    m.m[1][2] = (2 * (yz + wx)) * p.scale.y;
    m.m[2][0] = (2 * (xz + wy)) * p.scale.z;
    m.m[2][1] = (2 * (yz - wx)) * p.scale.z;
    m.m[2][2] = (1 - 2 * (xx + yy)) * p.scale.z;
    m.m[3][0] = p.translation.x;
    m.m[3][1] = p.translation.y;
    m.m[3][2] = p.translation.z;
    m.m[3][3] = 1.0f;
    return m;
}

void restPose(const Skeleton& skeleton, std::vector<JointPose>& out) {
    out.resize(skeleton.joints.size());
    for (size_t j = 0; j < out.size(); ++j) out[j] = skeleton.joints[j].rest;
}

void samplePose(const Skeleton& skeleton, const AnimationClip& clip, f32 seconds, bool loop, std::vector<JointPose>& out) {
    if (clip.frames <= 1 || !std::isfinite(seconds)) {
        framesBetween(skeleton, clip, 0, 0, 0.0f, out);
        return;
    }
    if (loop) seconds = std::fmod(std::fmod(seconds, clip.duration) + clip.duration, clip.duration);
    const f32 frame = std::clamp(seconds * static_cast<f32>(kClipFrameRate), 0.0f, static_cast<f32>(clip.frames - 1));
    const u32 f0 = static_cast<u32>(frame);
    const u32 f1 = std::min(f0 + 1, clip.frames - 1);
    framesBetween(skeleton, clip, f0, f1, frame - static_cast<f32>(f0), out);
}

void samplePoseTick(const Skeleton& skeleton, const AnimationClip& clip, u32 tick, u32 tickRate, bool loop, std::vector<JointPose>& out) {
    if (clip.frames <= 1 || tickRate == 0) {
        framesBetween(skeleton, clip, 0, 0, 0.0f, out);
        return;
    }
    // frame = tick × 30 / tick rate (integer division + remainder/tick rate: at 30/60 Hz only 0 and 0.5, so exact in float)
    const u64 scaled = static_cast<u64>(tick) * kClipFrameRate;
    const u64 last = clip.frames - 1;
    u64 f0 = scaled / tickRate;
    const u32 remainder = static_cast<u32>(scaled % tickRate);
    if (loop) {
        f0 %= last; // last frame = first frame (looping clip)
    } else if (f0 >= last) {
        framesBetween(skeleton, clip, static_cast<u32>(last), static_cast<u32>(last), 0.0f, out);
        return;
    }
    const f32 t = static_cast<f32>(remainder) / static_cast<f32>(tickRate);
    framesBetween(skeleton, clip, static_cast<u32>(f0), static_cast<u32>(f0 + 1), t, out);
}

void blendPoses(const std::vector<JointPose>& a, const std::vector<JointPose>& b, f32 weight, std::vector<JointPose>& out) {
    const size_t count = std::min(a.size(), b.size());
    out.resize(count);
    weight = std::clamp(weight, 0.0f, 1.0f);
    for (size_t j = 0; j < count; ++j) out[j] = lerpPose(a[j], b[j], weight);
}

void jointMatrices(const Skeleton& skeleton, const std::vector<JointPose>& pose, std::vector<Mat4>& out) {
    const size_t count = skeleton.joints.size();
    out.resize(count);
    for (size_t j = 0; j < count; ++j) {
        const Mat4 local = composeTrs(j < pose.size() ? pose[j] : skeleton.joints[j].rest);
        const i32 parent = skeleton.joints[j].parent;
        out[j] = local * (parent >= 0 ? out[static_cast<size_t>(parent)] : skeleton.root); // parents come first
    }
}

void skinPalette(const Skeleton& skeleton, const std::vector<JointPose>& pose, std::vector<BoneMatrix>& out) {
    std::vector<Mat4> world;
    jointMatrices(skeleton, pose, world);
    out.resize(world.size());
    for (size_t j = 0; j < world.size(); ++j) {
        const Mat4 skin = skeleton.joints[j].inverseBind * world[j]; // vertex: bind → joint → world (row vectors)
        // MealKit has z flipped: S·M·S (S = diag(1, 1, -1)). Shaders multiply with column vectors, so the three rows = columns of the row-vector matrix
        for (int r = 0; r < 3; ++r) {
            const f32 sr = r == 2 ? -1.0f : 1.0f;
            for (int c = 0; c < 4; ++c) {
                const f32 sc = c == 2 ? -1.0f : 1.0f;
                out[j].rows[r][c] = skin.m[c][r] * sr * sc; // c = 3 is translation (row 3)
            }
        }
    }
}

const AnimationClip* findClip(const std::vector<AnimationClip>& clips, const std::string& name) {
    for (const AnimationClip& c : clips)
        if (c.name == name) return &c;
    return nullptr;
}

} // namespace mealkit::asset
