#include <iostream>
#include <stdexcept>

#include "HelloTriangleApplication.hpp"

int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
