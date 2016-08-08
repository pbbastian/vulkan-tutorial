#pragma once

#include <memory>
#include <stb_image.h>

namespace pbbastian {
namespace stbi {
enum class comp_type {
  default_ = 0, // only used for req_comp
  grey = 1,
  grey_alpha = 2,
  rgb = 3,
  rgb_alpha = 4
};

using image = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;

inline image load(const char *filename, int &x, int &y, int &comp,
                  comp_type req_comp) {
  auto image_ptr =
      stbi_load(filename, &x, &y, &comp, static_cast<int>(req_comp));
  return image(image_ptr, stbi_image_free);
}

inline image load(const char *filename, int &x, int &y, comp_type req_comp) {
  auto image_ptr =
      stbi_load(filename, &x, &y, nullptr, static_cast<int>(req_comp));
  return image(image_ptr, stbi_image_free);
}
}
}
