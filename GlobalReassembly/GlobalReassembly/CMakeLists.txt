set(PROJECT_NAME GlobalReassembly)

################################################################################
# Source groups
################################################################################
set(no_group_source_files
    "Edge.h"
    "JigsawOpt2.cpp"
    "JigsawOpt2.h"
    "Loop.h"
    "main.cpp"
    "MultiGraph.cpp"
    "MultiGraph.h"
    "Parameters.cpp"
    "Parameters.h"
    "Utils.h"
)
source_group("" FILES ${no_group_source_files})

set(IntersectionDetector
    "IntersectionDetector/CUDAImage.h"
    "IntersectionDetector/CUDAImgIntersectionDetector.cpp"
    "IntersectionDetector/CUDAImgIntersectionDetector.h"
)
source_group("IntersectionDetector" FILES ${IntersectionDetector})

set(IntersectionDetector__CUDAImageTransform
    "IntersectionDetector/CUDAImageTransform/TransformImage.cpp"
    "IntersectionDetector/CUDAImageTransform/TransformImage.cu"
    "IntersectionDetector/CUDAImageTransform/TransformImage.cuh"
    "IntersectionDetector/CUDAImageTransform/TransformImage.h"
    "IntersectionDetector/CUDAImageTransform/Utils.cpp"
    "IntersectionDetector/CUDAImageTransform/Utils.h"
)
source_group("IntersectionDetector\\CUDAImageTransform" FILES ${IntersectionDetector__CUDAImageTransform})

set(JigsawVisual
    "JigsawVisual/JigsawVisual.cpp"
    "JigsawVisual/JigsawVisual.h"
    "JigsawVisual/Utils.h"
)
source_group("JigsawVisual" FILES ${JigsawVisual})

set(ALL_FILES
    ${no_group_source_files}
    ${IntersectionDetector}
    ${IntersectionDetector__CUDAImageTransform}
    ${JigsawVisual}
)

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})

# Check if CUDA is available and set properties for TransformImage.cu
if(CUDAToolkit_FOUND)
    # Set CUDA compilation for TransformImage.cu using modern CMake
    set_source_files_properties(IntersectionDetector/CUDAImageTransform/TransformImage.cu PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
    message(STATUS "CUDA support enabled for TransformImage.cu")
else()
    message(WARNING "CUDA not available - TransformImage.cu will not be compiled")
endif()

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
################################################################################
# Includes for CMake from *.props
################################################################################
use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "$ENV{VCTargetsPath}/BuildCustomizations/CUDA 9.0.cmake")

set(ROOT_NAMESPACE GlobalReassembly)

set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
)
################################################################################
# Compile definitions
################################################################################
target_compile_definitions(${PROJECT_NAME} PRIVATE
    "$<$<CONFIG:Debug>:"
        "_DEBUG"
    ">"
    "$<$<CONFIG:Release>:"
        "NDEBUG"
    ">"
    "WIN32;"
    "WIN64;"
    "_CONSOLE;"
    "_MBCS"
)

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:
            /Od
        >
        $<$<CONFIG:Release>:
            /O2;
            /Oi;
            /Gy;
            /openmp
        >
        /sdl-;
        /W3;
        ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
        /await;
        ${DEFAULT_CXX_EXCEPTION_HANDLING}
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:
            /INCREMENTAL
        >
        $<$<CONFIG:Release>:
            /OPT:REF;
            /OPT:ICF
        >
        /DEBUG;
        /SUBSYSTEM:CONSOLE
    )
endif()

if(UNIX)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:
            -g;
            -O0;
        >
        $<$<CONFIG:Release>:
            -O2;
        >
        -Wall;
        -march=x86-64;
        -fopenmp;
        $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++23>
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>
        -I/usr/include/opencv4
        -I/usr/include/eigen3
        -I/usr/local/cuda-13.0/targets/x86_64-linux/include
    )
    target_link_directories(${PROJECT_NAME} PRIVATE
        /usr/local/cuda-13.0/targets/x86_64-linux/lib
    )
endif()

################################################################################
# Post build events
################################################################################
# add_custom_command_if(
#     TARGET ${PROJECT_NAME}
#     POST_BUILD
#     COMMANDS
#     COMMAND   $<CONFIG:Debug> echo copy "$ENV{CudaToolkitBinDir}\\cudart*.dll" "$<SHELL_PATH:${OUTPUT_DIRECTORY}>"
#     COMMAND   $<CONFIG:Debug> copy "$ENV{CudaToolkitBinDir}\\cudart*.dll" "$<SHELL_PATH:${OUTPUT_DIRECTORY}>"
#     COMMAND $<CONFIG:Release> echo copy "$ENV{CudaToolkitBinDir}\\cudart*.dll" "$<SHELL_PATH:${OUTPUT_DIRECTORY}>"
#     COMMAND $<CONFIG:Release> copy "$ENV{CudaToolkitBinDir}\\cudart*.dll" "$<SHELL_PATH:${OUTPUT_DIRECTORY}>"
# )

################################################################################
# Dependencies
################################################################################
find_package(OpenMP)
set(ADDITIONAL_LIBRARY_DEPENDENCIES
    "opencv_core;"
    "opencv_imgcodecs;"
    "opencv_highgui;"
    "opencv_imgproc;"
    "cudart;"
    "OpenMP::OpenMP_CXX;"
    # "kernel32;"
    # "user32;"
    # "gdi32;"
    # "winspool;"
    # "comdlg32;"
    # "advapi32;"
    # "shell32;"
    # "ole32;"
    # "oleaut32;"
    # "uuid;"
    # "odbc32;"
    # "odbccp32"
)
target_link_libraries(${PROJECT_NAME} PRIVATE "${ADDITIONAL_LIBRARY_DEPENDENCIES}")

