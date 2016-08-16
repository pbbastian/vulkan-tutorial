#pragma once

#include <string>

namespace settings {

constexpr bool enableValidationLayers =
#ifdef LAVA_ENABLE_VALIDATION_LAYERS
    true
#else
    false
#endif
;

const std::string assetPath =
#ifdef ASSET_PATH
    ASSET_PATH
#else
    "./"
#endif
;

}
