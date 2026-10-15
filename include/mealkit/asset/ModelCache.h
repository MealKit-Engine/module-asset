#pragma once

#include "mealkit/core/Math.h"
#include "mealkit/render/Renderer.h"
#include "mealkit/scene/Scene.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// glTF(.glb/.gltf) 모델 불러오기와 캐시.
// 불러올 때 meshoptimizer로 정점 중복 제거, 캐시/오버드로/페치 최적화를 하고 LOD를 최대 2단계 더 만든다.
// 물리용 볼록 껍질 점(hullPoints)은 파일의 정점에서 바로 뽑아 모든 PC에서 같다.
namespace mealkit::asset {

inline constexpr u32 kMaxLods = 3;

struct Model {
    bool valid = false;
    std::string path; // 씬에 적힌 경로 (assets 기준 상대 경로)
    render::MeshId lods[kMaxLods] = {};
    u32 lodCount = 0;
    u32 triangles[kMaxLods] = {};
    Vec3 boundsMin, boundsMax;    // 모델 공간 (MealKit 좌표계: 왼손, y 위)
    std::vector<Vec3> hullPoints; // 물리 볼록 껍질용
    u64 fileHash = 0;             // 원본 파일 내용 해시 (멀티플레이 검사용)
};

class ModelCache {
public:
    // renderer가 없으면 CPU 데이터(경계, 껍질, 해시)만 만든다.
    void setRenderer(render::Renderer* renderer) { m_renderer = renderer; }

    // 처음 요청 때 불러오고 이후에는 캐시에서 준다. 실패하면 nullptr (같은 경로는 다시 시도하지 않음).
    const Model* get(const std::string& path);
    render::MeshId boxMesh(); // 한 변 1인 기본 상자

    // 화면에서 차지하는 크기로 LOD를 고른다.
    render::MeshId pickLod(const Model& model, Vec3 cameraPos, Vec3 position, Vec3 scale) const;

    // 씬이 참조하는 모든 모델 파일 내용의 해시 (scene.contentHash()와 합쳐 접속 검사에 쓴다).
    u64 hashModels(const scene::Scene& scene);

    static std::string resolve(const std::string& path); // assets 기준 경로 → 실제 파일 경로
    static std::string assetsRoot();                     // 저장소/실행 파일 근처의 assets 폴더

private:
    render::Renderer* m_renderer = nullptr;
    render::MeshId m_box = 0xFFFFFFFFu;
    std::unordered_map<std::string, std::unique_ptr<Model>> m_models;
};

} // namespace mealkit::asset
