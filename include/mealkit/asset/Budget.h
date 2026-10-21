#pragma once

#include "mealkit/core/Types.h"

#include <string>

// 에셋 예산 (docs/ROADMAP.md 2단계 "에셋 예산 경고", docs/CODING_MODEL.md "모델 예산 종류"). 모델 종류는 폴더로 정한다:
//   assets/models/characters/... = 캐릭터, models/environment/... = 배경, 그 밖 = 소품
// 한도는 mealkit.project(budget<종류><항목>)에 있고 기본값은 아래 표다 (최소 사양 내장 그래픽 기준, 벤치마크로 고친다).
// 넘으면 불러올 때 경고하고 에디터 인스펙터·Blender 애드온이 같은 값으로 보여 준다. 시뮬레이션과 무관하다.
namespace mealkit::asset {

enum class ModelKind : u8 { Character, Prop, Environment, Count };
inline constexpr u32 kModelKindCount = static_cast<u32>(ModelKind::Count);

struct ModelBudget {
    u32 triangles = 0;   // LOD 0
    u32 textureSize = 0; // 가장 긴 변 (픽셀)
    u32 bones = 0;
    u32 materials = 0;
};

inline constexpr ModelBudget kDefaultBudgets[kModelKindCount] = {
    {15000, 2048, 64, 4}, // 캐릭터
    {3000, 1024, 0, 2},   // 소품
    {20000, 2048, 0, 4},  // 배경
};

// 폴더 이름 = 종류 이름 ("characters", "props", "environment")
inline const char* modelKindFolder(ModelKind kind) {
    switch (kind) {
    case ModelKind::Character: return "characters";
    case ModelKind::Environment: return "environment";
    default: return "props";
    }
}

// assets 기준 모델 경로("models/characters/hero.glb")의 종류. 종류 폴더가 아니면 소품
inline ModelKind modelKind(const std::string& path) {
    std::string p = path;
    for (char& c : p)
        if (c == '\\') c = '/';
    const std::string prefix = "models/";
    const size_t start = p.rfind(prefix, 0) == 0 ? prefix.size() : 0;
    const size_t slash = p.find('/', start);
    if (slash == std::string::npos) return ModelKind::Prop;
    const std::string folder = p.substr(start, slash - start);
    for (u32 k = 0; k < kModelKindCount; ++k)
        if (folder == modelKindFolder(static_cast<ModelKind>(k))) return static_cast<ModelKind>(k);
    return ModelKind::Prop;
}

} // namespace mealkit::asset
