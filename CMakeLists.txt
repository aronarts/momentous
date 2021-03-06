cmake_minimum_required(VERSION 2.8)

project(momentous C CXX)

# Avoid source tree pollution
if (momentous_SOURCE_DIR STREQUAL momentous_BINARY_DIR)
  message(WARNING "In-source builds are not recommended. "
                  "Consider building in a separate directory, "
                  "to avoid polluting the source tree.")
endif()

# Add a sensible default build type
if (NOT CMAKE_BUILD_TYPE)
  message(WARNING "CMAKE_BUILD_TYPE is not defined; defaulting to Debug.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Build type: None Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
  add_definitions(-std=c++11)
endif()

find_package(OpenGL REQUIRED)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "No examples")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "No tests")
set(GLFW_INSTALL OFF CACHE BOOL "Don't install")

add_subdirectory(deps)

include_directories(
                    ${GLEW_SOURCE_DIR}
                    ${GLFW_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/deps/glm)

file(GLOB_RECURSE ALL_SRCS
	RELATIVE ${CMAKE_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
)

add_executable(momentous ${ALL_SRCS}) 

target_link_libraries(momentous ${GLFW_LIBRARIES})
target_link_libraries(momentous glfw)

add_definitions(-DGLEW_STATIC)
target_link_libraries(momentous ${GLEW_LIBRARIES})
target_link_libraries(momentous GLEW)

target_link_libraries(momentous ${OPENGL_LIBRARIES})