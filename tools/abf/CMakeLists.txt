# CMakeLists for abf_unwrap
# Builds against the already-compiled geogram (headless, in WSL2).
# Build:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j16

cmake_minimum_required(VERSION 3.5)
project(abf_unwrap CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Absolute path to the geogram checkout + its Release build tree.
set(GEOSRC  /root/abf_toolkit/geogram)
set(GEOBUILD ${GEOSRC}/build/Release)

add_executable(abf_unwrap abf_unwrap.cpp)

# Match geogram's FP flags (needed for correct exact-predicate behavior).
target_compile_options(abf_unwrap PRIVATE -frounding-math -ffp-contract=off -fopenmp -O2)

# Same feature defines the library was compiled with (keeps headers consistent).
target_compile_definitions(abf_unwrap PRIVATE
    GEOGRAM_USE_BUILTIN_DEPS
    GEOGRAM_VERSION=\"1.10.1-rc\"
    GEOGRAM_WITH_HLBFGS
    GEOGRAM_WITH_LUA
    GEOGRAM_WITH_PDEL
    GEOGRAM_WITH_TETGEN
    GEOGRAM_WITH_TRIANGLE
    GEO_DYNAMIC_LIBS
    NL_WITH_AMGCL
)

target_include_directories(abf_unwrap PRIVATE
    ${GEOSRC}/src/lib
    ${GEOSRC}/src/lib/geogram_gfx/third_party
    ${GEOBUILD}/src/lib
    ${GEOSRC}/src/lib/geogram/third_party/amgcl
)

target_link_directories(abf_unwrap PRIVATE ${GEOBUILD}/lib)
target_link_libraries(abf_unwrap PRIVATE geogram tbb pthread dl)
target_link_options(abf_unwrap PRIVATE "-Wl,-rpath,${GEOBUILD}/lib")
