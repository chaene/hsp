
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing Torch through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()
set(CMAKE_C_FLAGS "-std=c99")

FIND_PACKAGE(Torch REQUIRED)

SET(src hsp.cpp)
SET(luasrc init.lua)
ADD_TORCH_PACKAGE(hsp "${src}" "${luasrc}" "hierarchical surface prediction")
TARGET_LINK_LIBRARIES(hsp luaT TH X11)