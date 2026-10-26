#pragma once

#include "mealkit/core/Math.h"
#include "mealkit/render/Mesh.h"

#include <string>
#include <vector>

// 스켈레탈 애니메이션 (docs/ROADMAP.md 2단계 "스켈레탈 애니메이션"). glTF 스킨·애니메이션을 불러올 때(asset::ModelCache) 여기로 옮긴다.
// 클립은 kClipFrameRate(30 Hz)로 미리 구운 관절 TRS라 재생은 두 프레임 사이 보간뿐이다.
// 좌표: 관절 값·역바인드는 glTF 공간(오른손) 그대로 두고, 화면용 팔레트를 만들 때 z를 뒤집어 MealKit(왼손)로 바꾼다.
//
// 두 가지 재생 길:
//   - 화면용 samplePose(clip, 초): 실수 시간, view/만. 결과가 시뮬레이션에 들어가면 안 된다 (히트박스·판정은 sim의 캡슐)
//   - 시뮬레이션용 samplePoseTick(clip, 틱, 틱레이트): 정수 틱 → 정수 프레임 + 정확히 나타나는 분수, 사칙연산 lerp·nlerp만.
//     액티브 래그돌 목표 포즈처럼 sim/이 써야 하는 포즈 (docs/DECISIONS.md "액티브 래그돌"). 화면용 포즈와 섞지 않는다
namespace mealkit::asset {

inline constexpr u32 kClipFrameRate = 30; // 구운 클립의 프레임 (Hz)
inline constexpr u32 kMaxJoints = 64;     // GPU 팔레트 한도 (넘는 스킨은 불러올 때 거부)

struct JointPose {
    Vec3 translation;
    Quat rotation;
    Vec3 scale{1, 1, 1};
};

struct Skeleton {
    struct Joint {
        std::string name;
        i32 parent = -1;     // joints 안 번호 (부모가 앞에 온다), -1 = 뿌리
        JointPose rest;      // 쉬는 자세 (애니메이션이 없는 관절은 이 값)
        Mat4 inverseBind;    // glTF 역바인드 행렬 (행 벡터 규약으로 옮긴 것)
    };
    std::vector<Joint> joints;
    Mat4 root = Mat4::identity(); // 뿌리 관절 위의 노드 변환 (glTF 공간, 행 벡터)
    bool valid() const { return !joints.empty(); }
};

struct AnimationClip {
    std::string name;
    u32 frames = 0;               // 1 이상 (마지막 프레임 = 길이)
    f32 duration = 0.0f;          // 초 = (frames - 1) / kClipFrameRate
    std::vector<JointPose> poses; // frames × 관절 수
    const JointPose* frame(u32 f, u32 jointCount) const { return &poses[static_cast<size_t>(f) * jointCount]; }
};

using render::BoneMatrix; // 화면용 팔레트 한 칸 (render/Mesh.h)

// 클립의 시간(초) 포즈. loop면 길이로 감고, 아니면 끝에서 멈춘다. out은 관절 수만큼
void samplePose(const Skeleton& skeleton, const AnimationClip& clip, f32 seconds, bool loop, std::vector<JointPose>& out);
// 결정론 (sim/ 전용): 틱 번호와 틱레이트로. 같은 입력이면 모든 PC에서 비트까지 같다
void samplePoseTick(const Skeleton& skeleton, const AnimationClip& clip, u32 tick, u32 tickRate, bool loop, std::vector<JointPose>& out);
// 두 포즈를 섞는다 (weight 0 = a, 1 = b). 크로스페이드
void blendPoses(const std::vector<JointPose>& a, const std::vector<JointPose>& b, f32 weight, std::vector<JointPose>& out);
// 쉬는 자세
void restPose(const Skeleton& skeleton, std::vector<JointPose>& out);
// 관절 월드 행렬 (glTF 공간, 행 벡터, 뿌리 노드 변환 포함)
void jointMatrices(const Skeleton& skeleton, const std::vector<JointPose>& pose, std::vector<Mat4>& out);
// 화면용 팔레트: 관절 월드 × 역바인드, MealKit 좌표(z 반전)로. 렌더러 drawSkinned에 그대로 넘긴다
void skinPalette(const Skeleton& skeleton, const std::vector<JointPose>& pose, std::vector<BoneMatrix>& out);

Mat4 composeTrs(const JointPose& pose); // 행 벡터 규약 (v * M = 크기 → 회전 → 이동)
const AnimationClip* findClip(const std::vector<AnimationClip>& clips, const std::string& name); // 없으면 nullptr

} // namespace mealkit::asset
