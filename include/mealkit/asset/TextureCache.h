#pragma once

#include "mealkit/rhi/Rhi.h"

#include <string>
#include <vector>

// 모델 텍스처 캐시 (docs/ROADMAP.md 2단계 "텍스처와 머티리얼", docs/DECISIONS.md 2026-10-21 "텍스처 캐시").
// glTF 안의 PNG·JPEG를 밉맵과 함께 BC7(색·금속/거칠기·발광) 또는 BC5(법선)로 압축해 파일로 남긴다:
//   <프로젝트>/build/cache/textures/<내용 해시>.mktex   (git 밖, 지우면 다시 만든다)
// 압축은 개발 빌드(디버그 UI가 있는 빌드: 에디터·개발 exe)만 한다. 출시 빌드는 압축된 캐시만 읽는다
// (Export가 exe 옆 cache/textures로 복사한다). 화면 전용이라 시뮬레이션·콘텐츠 해시와 무관하다
// (원본 이미지는 glTF 파일 해시에 이미 들어 있다).
namespace mealkit::asset {

enum class TextureUsage : u8 {
    Color,  // baseColor·emissive: sRGB BC7
    Linear, // metallicRoughness: BC7 (선형)
    Normal, // 법선 맵: BC5 (x, y. 셰이더가 z를 다시 만든다)
};

struct CompressedTexture {
    u32 width = 0, height = 0, mipLevels = 0;
    rhi::Format format = rhi::Format::Bc7Unorm;
    std::vector<u8> data; // 밉 0부터 rhi::textureMipBytes씩 (rhi::TextureInit)
};

// 인코딩된 이미지(glTF의 PNG·JPEG 바이트) → 캐시에서 읽거나, 없으면 디코드·밉맵·압축하고 캐시에 쓴다. 실패하면 false와 이유
bool loadCompressedTexture(const void* encoded, size_t bytes, TextureUsage usage, CompressedTexture& out, std::string& error);

// 캐시 폴더: 쓰는 곳은 <프로젝트>/build/cache/textures, 읽을 때는 그다음 exe 옆 cache/textures (출시본)
std::string textureCacheFolder();

// 압축 없이 밉맵만 (단위 테스트용, RGBA8). srgb면 선형으로 바꿔 평균한다
void buildMips(const u8* rgba, u32 width, u32 height, bool srgb, bool normal, std::vector<std::vector<u8>>& mips);

} // namespace mealkit::asset
