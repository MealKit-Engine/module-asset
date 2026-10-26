#pragma once

#include "mealkit/core/Math.h"

#include <string>
#include <vector>

// 하늘 HDRI → 환경광·반사 (docs/ROADMAP.md 2단계 "하늘 HDRI와 환경광", docs/DECISIONS.md 2026-10-26 HDRI).
// 씬 환경의 `.hdr`(정거리원통 Radiance) 하나에서 불러올 때 굽고 파일로 남긴다:
//   <프로젝트>/build/cache/environments/<내용 해시>.mkenv   (git 밖, 지우면 다시 굽는다)
//   - 하늘 큐브맵 (스카이박스, 밉 하나)
//   - 반사 큐브맵 128² + 밉 (밉 k = 거칠기 k / (밉 수 − 1), GGX 중요도 표본)
//   - 환경광 구면 조화 9계수 (Lambert 합성곱까지 곱해 셰이더가 법선으로 바로 "환경광 색"을 얻는다)
// 큐브 면 순서·방향은 DX12/Vulkan 규칙(+X, −X, +Y, −Y, +Z, −Z, MealKit 왼손 y 위)이고 텍셀은 RGB9E5.
// 굽기는 개발 빌드(디버그 UI가 있는 빌드)만 한다: 출시 빌드는 캐시만 읽는다 (Export가 exe 옆 cache/environments로 복사).
// 화면 전용이라 시뮬레이션·콘텐츠 해시와 무관하다.
namespace mealkit::asset {

struct EnvironmentMap {
    f32 sh[9][3] = {}; // 환경광 색 = Σ sh[i] · Y_i(n) (Y: 실수 구면 조화 0~2차)
    u32 skySize = 0;   // 하늘 큐브 면 크기 (밉 하나)
    std::vector<u32> sky;
    u32 reflectSize = 0, reflectMips = 0; // 반사 큐브 (면마다 밉 0부터)
    std::vector<u32> reflect;
};

// hdrPath(.hdr 파일) → 캐시에서 읽거나, 없으면(개발 빌드) 굽고 캐시에 쓴다. 실패하면 false와 이유
bool loadEnvironment(const std::string& hdrPath, EnvironmentMap& out, std::string& error);

// assets/textures/ 아래(하위 폴더 포함)의 .hdr 파일 (assets 기준 경로 "textures/hdri/x.hdr", 이름 순). 에디터 [환경] 창·룩데브의 고르기 목록
std::vector<std::string> listHdris(const std::string& assetsRoot);

// 캐시 폴더: 쓰는 곳은 <프로젝트>/build/cache/environments, 읽을 때는 그다음 exe 옆 cache/environments (출시본)
std::string environmentCacheFolder();

// --- 굽기 부품 (단위 테스트)
u32 packRgb9e5(f32 r, f32 g, f32 b); // 음수는 0, 너무 큰 값은 최대로
void unpackRgb9e5(u32 v, f32 out[3]);
Vec3 cubeDirection(u32 face, f32 s, f32 t); // face 0..5, s·t = −1..1 (텍셀 가운데) → 정규화한 방향
// 정거리원통 그림(rgb float, width × height)에서 굽는다 (파일·캐시 없이)
void bakeEnvironment(const f32* rgb, u32 width, u32 height, EnvironmentMap& out);
Vec3 evalSh(const f32 (&sh)[9][3], Vec3 n); // 셰이더와 같은 계산

} // namespace mealkit::asset
