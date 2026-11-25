#pragma once

#include "mealkit/asset/Animation.h"
#include "mealkit/core/Reflect.h"
#include "mealkit/sim/Table.h"

#include <string>
#include <string_view>
#include <vector>

// Two-bone IK, root motion and motion warping (docs/DECISIONS.md 2026-11-25 "Animation extras"). View side only: results never enter the simulation.
// All positions here are in skeleton space (glTF space below Skeleton::root; the model's own space, right-handed). Game view code converts
// terrain points into it (inverse of the model matrix, then the z flip) before giving targets.
//
// assets/tables/ik.table, one row per chain. Row id = "<set>.<chain>" (set = the animation set / character kind).
//   root, mid, tip   joint names (thigh, knee, ankle; upper arm, elbow, wrist)
//   pole             "x y z" direction the middle joint bends towards (knee forward); empty = keep the animated bend
//   weight           default blend 0..1 (view code can override per frame)
//   ankle            height of the tip joint above the sole in skeleton units (feet on terrain)
namespace mealkit::asset {

MEALKIT_STRUCT(IkChain) {
    MEALKIT_FIELD(std::string, root, {}, "Root joint name (hip, shoulder)");
    MEALKIT_FIELD(std::string, mid, {}, "Middle joint name (knee, elbow)");
    MEALKIT_FIELD(std::string, tip, {}, "Tip joint name (ankle, wrist)");
    MEALKIT_FIELD(std::string, pole, {}, "Bend direction \"x y z\" (empty: keep the animated bend)");
    MEALKIT_FIELD(f32, weight, 1.0f, "Default weight (0..1)");
    MEALKIT_FIELD(f32, ankle, 0.0f, "Tip joint height above the sole");
};
inline constexpr const char* kIkTable = "tables/ik.table"; // relative to assets

// Chain with resolved joints
struct IkJoints {
    i32 root = -1, mid = -1, tip = -1;
    Vec3 pole{};
    bool hasPole = false;
    f32 weight = 1.0f;
    f32 ankle = 0.0f;
    bool valid() const { return root >= 0 && mid >= 0 && tip >= 0; }
};

class IkChains {
public:
    // Parses rows (warnings: bad ids, bad pole)
    void build(const sim::Table<IkChain>& table, std::string* warnings = nullptr);
    // Reads assets/tables/ik.table (false if missing)
    bool load();

    // Chain of a set by name resolved against a skeleton (valid() is false when the set, chain or a joint is missing)
    IkJoints find(std::string_view set, std::string_view chain, const Skeleton& skeleton) const;
    // Chain names of a set, in table order
    std::vector<std::string> names(std::string_view set) const;

private:
    struct Entry {
        std::string set, name;
        IkChain row;
        Vec3 pole{};
        bool hasPole = false;
    };
    std::vector<Entry> m_entries;
};

i32 findJoint(const Skeleton& skeleton, std::string_view name); // -1 if missing

// Position of a joint in skeleton space (rotation/translation only; joint scale is assumed to be 1)
Vec3 jointPosition(const Skeleton& skeleton, const std::vector<JointPose>& pose, i32 joint);

// Moves the tip of a chain to target by rotating the root and middle joints. weight 0 leaves the pose, 1 reaches the target.
// Out of reach targets stretch the chain straight towards the target. Returns true when the tip reached the target
bool solveTwoBone(const Skeleton& skeleton, const IkJoints& chain, Vec3 target, f32 weight, std::vector<JointPose>& pose);

// Lowers or raises a joint (usually the pelvis/hip) by dy along skeleton Y, in the parent's space. Use with the lowest foot's offset so the
// legs can reach the terrain
void offsetJoint(const Skeleton& skeleton, i32 joint, f32 dy, std::vector<JointPose>& pose);

// Foot on terrain: tip target = the current tip position with its height set to groundY + chain.ankle, weight scaled by how much the sole is
// above/below the ground is left to the caller. Returns the target; the height change (target.y - current.y) tells how far the foot moves
Vec3 groundTarget(const Skeleton& skeleton, const IkJoints& chain, const std::vector<JointPose>& pose, f32 groundY);

// Root motion --------------------------------------------------------------------------------------------------------------------------------
// Root travel of a clip (translation of the root joint, skeleton space) between two times in seconds. With loop the interval may wrap
Vec3 clipRootDelta(const Skeleton& skeleton, const AnimationClip& clip, i32 rootJoint, f32 fromSeconds, f32 toSeconds, bool loop);
// Total root travel of the clip from first to last frame
Vec3 clipRootTravel(const Skeleton& skeleton, const AnimationClip& clip, i32 rootJoint);
// Removes the root joint's horizontal (x, z) translation from the pose so the character stays on the simulation position; the removed part
// relative to the rest pose is returned. Vertical movement stays
Vec3 stripRootMotion(const Skeleton& skeleton, i32 rootJoint, std::vector<JointPose>& pose);

// Visual offset of the model from the simulation position, driven by root motion deltas and easing back to 0
struct RootOffset {
    Vec3 offset{};
    void add(Vec3 delta) { offset += delta; }
    // Pulls the offset back towards 0 at returnSpeed (units per second, at least a share of the offset so it never lingers)
    void update(f32 dt, f32 returnSpeed);
};

// Motion warping: scales and turns root travel so a clip whose travel is `travel` ends at `target` (both horizontal, skeleton space)
struct MotionWarp {
    f32 scale = 1.0f;
    f32 yaw = 0.0f; // radians around Y
    bool active = false;

    static MotionWarp make(Vec3 travel, Vec3 target, f32 maxScale = 2.0f);
    Vec3 apply(Vec3 delta) const;
};

} // namespace mealkit::asset
