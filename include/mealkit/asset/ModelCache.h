#pragma once

#include "mealkit/asset/Animation.h"
#include "mealkit/asset/Budget.h"
#include "mealkit/asset/Environment.h"
#include "mealkit/asset/TextureCache.h"
#include "mealkit/core/Math.h"
#include "mealkit/render/DrawList.h"
#include "mealkit/render/Renderer.h"
#include "mealkit/scene/Scene.h"

#include <algorithm>
#include <cstring>
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
    // 에셋 예산 (asset/Budget.h): 종류는 경로의 폴더, 사용량은 불러올 때 센다. 넘은 항목은 budgetProblems (비면 예산 안)
    ModelKind kind = ModelKind::Prop;
    u32 materials = 0;
    u32 bones = 0;       // 스킨의 관절 수 (kMaxJoints를 넘으면 스킨을 버리고 정적 모델로)
    // 스켈레탈 애니메이션 (asset/Animation.h): 스킨이 있으면 lods[0] 하나가 스킨 메시 (LOD 없음), clips는 glTF 애니메이션
    Skeleton skeleton;
    std::vector<AnimationClip> clips;
    bool skinned() const { return skeleton.valid(); }
    u32 textureSize = 0; // 가장 큰 텍스처의 긴 변 (렌더러가 있을 때)
    u32 textures = 0;
    std::vector<std::string> budgetProblems;
};

// 모델 하나를 목록에 넣는다: 정적 모델은 add, 스킨 모델은 anim("<클립> [초 어긋남]", 씬의 kAnimField)을 seconds에서
// 반복 재생한 팔레트로 addSkinned (클립이 없거나 비면 쉬는 자세). mesh는 pickLod 결과 (스킨 모델은 lods[0])
void addModel(render::DrawList& out, const Model& model, render::MeshId mesh, const render::InstanceData& instance,
              const std::string& anim = {}, f64 seconds = 0.0);

// 모델이 종류의 예산을 넘은 항목 ("15200 triangles (budget 15000)"). 영어 (로그·인스펙터)
std::vector<std::string> budgetProblems(const Model& model, const ModelBudget& budget);

// 모델 지원 범위를 외부 도구에 알리는 스키마 (JSON). Blender 애드온(tools/blender-addon/mealkit/schema.json)이 읽고,
// 단위 테스트 asset_schema가 그 파일이 이것과 같은지 본다 (엔진과 애드온이 어긋나지 않게). 불러오기가 바뀌면 같이 고친다
std::string modelSchemaJson();

// glTF 머티리얼 extras(JSON 객체)의 "mealkit_<키>"를 스타일 프리셋 머티리얼 기능으로 (스키마 materialFeatures, Blender 애드온이
// 커스텀 속성으로 쓴다). 큰 기능 키(outline, wind, rim, dissolve)가 있으면 그 기능을 켜고, 색 키는 그 기능의 값만 바꾼다.
// 모르는 mealkit_ 키·틀린 값은 problems에 영어로 (불러오기가 파일 이름과 함께 경고). mealkit_로 시작하지 않는 키는 보지 않는다
void applyMaterialExtras(const char* json, render::MaterialData& material, std::vector<std::string>& problems);

class ModelCache {
public:
    // renderer가 없으면 CPU 데이터(경계, 껍질, 해시)만 만든다 (헤드리스).
    // 렌더러 호출은 이 인라인 함수 안에만 있어, 부르지 않는 헤드리스 빌드는 렌더러 없이 링크된다.
    void setRenderer(render::Renderer* renderer) {
        m_uploadContext = renderer;
        m_upload = renderer ? [](void* r, const render::MeshData& mesh) {
            return static_cast<render::Renderer*>(r)->createMesh(mesh);
        } : nullptr;
        m_uploadMaterial = renderer ? [](void* r, const render::MaterialData& m) {
            return static_cast<render::Renderer*>(r)->createMaterial(m);
        } : nullptr;
        // 텍스처: 캐시에서 읽거나 압축해 올린다 (asset/TextureCache.cpp, 화면 쪽). 실패하면 kNoTexture와 이유
        m_uploadTexture = renderer ? [](void* r, const void* encoded, size_t bytes, TextureUsage usage, u32* size,
                                        std::string& error) {
            CompressedTexture t;
            if (!loadCompressedTexture(encoded, bytes, usage, t, error)) return render::kNoTexture;
            *size = std::max(t.width, t.height);
            return static_cast<render::Renderer*>(r)->createTexture({t.width, t.height, t.mipLevels, t.format, t.data.data()});
        } : nullptr;
        // 하늘 HDRI: 캐시에서 읽거나 굽고 올린다 (asset/Environment.cpp, 화면 쪽). 실패하면 0과 이유
        m_uploadEnvironment = renderer ? [](void* r, const std::string& file, std::string& error) -> u32 {
            EnvironmentMap e;
            if (!loadEnvironment(file, e, error)) return 0;
            render::EnvironmentData d;
            std::memcpy(d.sh, e.sh, sizeof(d.sh));
            d.skySize = e.skySize;
            d.sky = e.sky.data();
            d.reflectSize = e.reflectSize;
            d.reflectMips = e.reflectMips;
            d.reflect = e.reflect.data();
            return static_cast<render::Renderer*>(r)->createEnvironment(d);
        } : nullptr;
        // 반사 프로브: assets/probes/의 구운 파일을 올린다. 실패하면 0과 이유
        m_uploadProbe = renderer ? [](void* r, const std::string& file, std::string& error) -> u32 {
            ProbeMap probe;
            if (!loadProbe(file, probe, error)) return 0;
            return static_cast<render::Renderer*>(r)->createProbe(probe.size, probe.mips, probe.reflect.data());
        } : nullptr;
        // 색 보정 LUT: .cube를 읽어 올린다. 실패하면 0과 이유
        m_uploadLut = renderer ? [](void* r, const std::string& file, std::string& error) -> u32 {
            std::vector<f32> rgb;
            if (!loadCubeLut(file, rgb, error)) return 0;
            return static_cast<render::Renderer*>(r)->createLut(rgb.data());
        } : nullptr;
    }

    // 처음 요청 때 불러오고 이후에는 캐시에서 준다. 실패하면 nullptr (같은 경로는 다시 시도하지 않음).
    const Model* get(const std::string& path);
    render::MeshId boxMesh(); // 한 변 1인 기본 상자
    // 하늘 HDRI (씬 환경의 hdri, assets 기준 경로) → 렌더러 환경 번호 (FrameParams::environment). 처음 요청 때 캐시에서 읽거나
    // 굽고(개발 빌드) 이후에는 번호만 준다. 비었거나 렌더러가 없거나 실패하면 0 (그라디언트, 같은 경로는 한 번만 경고).
    // reloadChanged가 파일이 바뀐 것을 보면 다음 요청 때 다시 굽는다 (옛 GPU 텍스처는 남는다)
    u32 environment(const std::string& hdriPath);
    // 반사 프로브 파일 (assets 기준 .mkprobe) → 렌더러 프로브 번호. environment와 같이 처음 요청 때 올리고, 바뀐 파일은 다시 올린다
    u32 probe(const std::string& probePath);
    // 색 보정 LUT (씬 환경 post.lut, assets 기준 .cube) → 렌더러 LUT 번호 (PostParams::lut). environment와 같이 처음 요청 때 올리고,
    // 바뀐 파일은 다시 올린다. 비었거나 실패하면 0 (보정 없음)
    u32 lut(const std::string& lutPath);
    // 씬 환경의 포스트 값 → FrameParams::post (LUT 파일은 lut()로 번호)
    render::PostParams postParams(const scene::PostFields& fields);
    // 씬의 반사 프로브 물체(scene::kTypeProbe, 구운 것만) → FrameParams::probes. 위치는 월드 좌표 (계층·펼친 씬 모두).
    // 한도(setProbeLimit, 프로젝트 설정 probeMaxCount)를 넘는 프로브는 버린다
    void probes(const scene::Scene& scene, std::vector<render::ReflectionProbe>& out);
    void setProbeLimit(u32 count) { m_probeLimit = std::clamp<u32>(count, 1, render::kMaxReflectionProbes); }

    // 에셋 핫 리로드 (에디터 편집 모드): 요청된 적 있는 모델 파일 중 바뀐 것(없던 파일이 생긴 것 포함)을 다시 읽는다.
    // Model 주소는 그대로 두고 내용만 바꾼다. 다시 읽은 경로 목록. 실패하면 이전 내용을 그대로 둔다.
    // 옛 GPU 메시는 지우지 않는다 (렌더러에 메시 지우기가 없다, 개발 중에만 쌓인다)
    std::vector<std::string> reloadChanged();

    // 종류별 예산 (mealkit.project, game::modelBudgets). 이미 불러온 모델도 다시 검사한다
    void setBudgets(const ModelBudget (&budgets)[kModelKindCount]);
    const ModelBudget& budget(ModelKind kind) const { return m_budgets[static_cast<u32>(kind)]; }

    // 화면에서 차지하는 크기로 LOD를 고른다.
    render::MeshId pickLod(const Model& model, Vec3 cameraPos, Vec3 position, Vec3 scale) const;

    // 씬이 참조하는 모든 모델 파일 내용의 해시 (scene.contentHash()와 합쳐 접속 검사에 쓴다).
    u64 hashModels(const scene::Scene& scene);

    static std::string resolve(const std::string& path); // assets 기준 경로 → 실제 파일 경로
    static std::string assetsRoot();                     // 저장소/실행 파일 근처의 assets 폴더

private:
    bool load(const std::string& path, Model& out);
    render::MeshId uploadMesh(const render::MeshData& mesh) { return m_upload(m_uploadContext, mesh); }

    render::MeshId (*m_upload)(void*, const render::MeshData&) = nullptr;
    u32 (*m_uploadMaterial)(void*, const render::MaterialData&) = nullptr;
    u32 (*m_uploadTexture)(void*, const void*, size_t, TextureUsage, u32*, std::string&) = nullptr;
    u32 (*m_uploadEnvironment)(void*, const std::string&, std::string&) = nullptr;
    u32 (*m_uploadProbe)(void*, const std::string&, std::string&) = nullptr;
    u32 (*m_uploadLut)(void*, const std::string&, std::string&) = nullptr;
    void* m_uploadContext = nullptr;
    struct EnvironmentEntry {
        u32 id = 0; // 0 = 실패 (다시 시도하지 않는다)
        unsigned long long time = 0;
    };
    std::unordered_map<std::string, EnvironmentEntry> m_environments;
    std::unordered_map<std::string, EnvironmentEntry> m_probes;
    std::unordered_map<std::string, EnvironmentEntry> m_luts;
    u32 m_probeLimit = 8;
    render::MeshId m_box = 0xFFFFFFFFu;
    std::unordered_map<std::string, std::unique_ptr<Model>> m_models;
    std::unordered_map<std::string, unsigned long long> m_fileTimes;
    ModelBudget m_budgets[kModelKindCount] = {kDefaultBudgets[0], kDefaultBudgets[1], kDefaultBudgets[2]};
    void checkBudget(Model& model) const; // 불러올 때 파일 시각 (reloadChanged)
};

} // namespace mealkit::asset
