#pragma once

#include "mealkit/core/Math.h"
#include "mealkit/asset/MeshData.h"

#include <string>
#include <vector>

// Skeletal animation (docs/ROADMAP.md stage 2 "Skeletal animation"). glTF skins/animations are moved here on load (asset::ModelCache).
// Clips are joint TRS pre-baked at kClipFrameRate (30 Hz), so playback is just interpolation between two frames.
// Coordinates: joint values and inverse binds stay in glTF space (right-handed); z is flipped to MealKit (left-handed) when building the view palette.
//
// Two playback paths:
//   - view samplePose(clip, seconds): real-valued time, view/ only. The result must not enter the simulation (hitboxes/checks use sim capsules)
//   - simulation samplePoseTick(clip, tick, tickRate): integer tick → integer frame + an exactly representable fraction, arithmetic-only lerp/nlerp.
//     For poses sim/ must use, like active ragdoll target poses (docs/DECISIONS.md "Active ragdoll"). Not mixed with view poses
namespace mealkit::asset {

inline constexpr u32 kClipFrameRate = 30; // frames of baked clips (Hz)
inline constexpr u32 kMaxJoints = 64;     // GPU palette limit (skins over this are rejected on load)

struct JointPose {
    Vec3 translation;
    Quat rotation;
    Vec3 scale{1, 1, 1};
};

struct Skeleton {
    struct Joint {
        std::string name;
        i32 parent = -1;     // index within joints (parents come first), -1 = root
        JointPose rest;      // rest pose (joints without animation use this)
        Mat4 inverseBind;    // glTF inverse bind matrix (converted to the row vector convention)
    };
    std::vector<Joint> joints;
    Mat4 root = Mat4::identity(); // node transform above the root joint (glTF space, row vectors)
    bool valid() const { return !joints.empty(); }
};

struct AnimationClip {
    std::string name;
    u32 frames = 0;               // at least 1 (last frame = length)
    f32 duration = 0.0f;          // seconds = (frames - 1) / kClipFrameRate
    std::vector<JointPose> poses; // frames × joint count
    const JointPose* frame(u32 f, u32 jointCount) const { return &poses[static_cast<size_t>(f) * jointCount]; }
};


// Pose of the clip at a time (seconds). With loop, wraps by length; otherwise stops at the end. out has one entry per joint
void samplePose(const Skeleton& skeleton, const AnimationClip& clip, f32 seconds, bool loop, std::vector<JointPose>& out);
// Deterministic (sim/ only): by tick number and tick rate. Same input gives bit-identical results on every PC
void samplePoseTick(const Skeleton& skeleton, const AnimationClip& clip, u32 tick, u32 tickRate, bool loop, std::vector<JointPose>& out);
// Blend two poses (weight 0 = a, 1 = b). Crossfade
void blendPoses(const std::vector<JointPose>& a, const std::vector<JointPose>& b, f32 weight, std::vector<JointPose>& out);
// Rest pose
void restPose(const Skeleton& skeleton, std::vector<JointPose>& out);
// Joint world matrices (glTF space, row vectors, including the root node transform)
void jointMatrices(const Skeleton& skeleton, const std::vector<JointPose>& pose, std::vector<Mat4>& out);
// View palette: joint world × inverse bind, in MealKit coordinates (z flipped). Passed directly to the renderer's drawSkinned
void skinPalette(const Skeleton& skeleton, const std::vector<JointPose>& pose, std::vector<BoneMatrix>& out);

Mat4 composeTrs(const JointPose& pose); // row vector convention (v * M = scale → rotate → translate)
const AnimationClip* findClip(const std::vector<AnimationClip>& clips, const std::string& name); // nullptr if not found

} // namespace mealkit::asset
