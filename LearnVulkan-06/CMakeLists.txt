# Function for building single example
function(buildProject PROJECT_NAME)
	SET(EXAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME})
	message(STATUS "Generating project file for example in ${EXAMPLE_FOLDER}")
	# Main
	# file(GLOB SOURCE *.cpp ${BASE_HEADERS} ${EXAMPLE_FOLDER}/*.cpp)
	SET(MAIN_CPP ${EXAMPLE_FOLDER}/${PROJECT_NAME}.cpp)
	if(EXISTS ${EXAMPLE_FOLDER}/main.cpp)
		SET(MAIN_CPP ${EXAMPLE_FOLDER}/main.cpp)
	endif()
	if(EXISTS ${EXAMPLE_FOLDER}/${PROJECT_NAME}.h)
		SET(MAIN_HEADER ${EXAMPLE_FOLDER}/${PROJECT_NAME}.h)
	endif()
	if(WIN32)
		add_executable(${PROJECT_NAME} WIN32 ${MAIN_CPP})
		target_link_libraries(${PROJECT_NAME} ${Vulkan_LIBRARY} ${WINLIBS})
	else(WIN32)
		add_executable(${PROJECT_NAME} ${MAIN_CPP})
		target_link_libraries(${PROJECT_NAME})
	endif(WIN32)

    set (CURRENT_WORKING_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	set (SHADERS_DIR ${CURRENT_WORKING_DIR}/ShaderCaches)
	set (TEXTURES_DIR ${CURRENT_WORKING_DIR}/Textures)
	set (MODELS_DIR ${CURRENT_WORKING_DIR}/Models)
	set (SHADERS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/shaders)
	set (TEXTURES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/textures)
	set (MODELS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/models)

	# Compile shader and copy texures
	add_custom_command(
    OUTPUT SHADER_COMPILE
	COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    COMMAND glslc ARGS ${SHADERS_SRC}/${PROJECT_NAME}.frag -o ${SHADERS_DIR}/${PROJECT_NAME}_frag.spv
    COMMAND glslc ARGS ${SHADERS_SRC}/${PROJECT_NAME}.vert -o ${SHADERS_DIR}/${PROJECT_NAME}_vert.spv
    COMMAND glslc ARGS ${SHADERS_SRC}/${PROJECT_NAME}_bg.frag -o ${SHADERS_DIR}/${PROJECT_NAME}_bg_frag.spv
    COMMAND glslc ARGS ${SHADERS_SRC}/${PROJECT_NAME}_bg.vert -o ${SHADERS_DIR}/${PROJECT_NAME}_bg_vert.spv
	WORKING_DIRECTORY ${SHADERS_SRC}
    DEPENDS ${SHADERS_SRC} ${SHADER_SOURCES}
	VERBATIM
	COMMENT "Compiling Shaders!"
    COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/background.png ${TEXTURES_DIR}/background.png
    COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/hylian_shield_c.png ${TEXTURES_DIR}/hylian_shield_c.png
    COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/hylian_shield_m.png ${TEXTURES_DIR}/hylian_shield_m.png
    COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/hylian_shield_r.png ${TEXTURES_DIR}/hylian_shield_r.png
    COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/hylian_shield_n.png ${TEXTURES_DIR}/hylian_shield_n.png
    COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/hylian_shield_o.png ${TEXTURES_DIR}/hylian_shield_o.png
    COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/master_sword_c.png ${TEXTURES_DIR}/master_sword_c.png
    COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/master_sword_m.png ${TEXTURES_DIR}/master_sword_m.png
    COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/master_sword_r.png ${TEXTURES_DIR}/master_sword_r.png
    COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/master_sword_n.png ${TEXTURES_DIR}/master_sword_n.png
    COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/master_sword_o.png ${TEXTURES_DIR}/master_sword_o.png
	COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/steath_c.png ${TEXTURES_DIR}/steath_c.png
	COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/steath_m.png ${TEXTURES_DIR}/steath_m.png
	COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/steath_r.png ${TEXTURES_DIR}/steath_r.png
	COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/steath_n.png ${TEXTURES_DIR}/steath_n.png
	COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURES_SRC}/steath_o.png ${TEXTURES_DIR}/steath_o.png
	COMMENT "Copying Textures!"
    COMMAND ${CMAKE_COMMAND} -E copy ${MODELS_SRC}/hylian_shield.obj ${MODELS_DIR}/hylian_shield.obj
    COMMAND ${CMAKE_COMMAND} -E copy ${MODELS_SRC}/master_sword.obj ${MODELS_DIR}/master_sword.obj
    COMMAND ${CMAKE_COMMAND} -E copy ${MODELS_SRC}/steath.obj ${MODELS_DIR}/steath.obj
	COMMENT "Copying Models!"
    )

	set(COMPILE_SHADER_TARGET ${PROJECT_NAME}_shader)
	set(SHADER_SOURCES ${SHADERS_SRC}/${PROJECT_NAME}.frag ${SHADERS_SRC}/${PROJECT_NAME}.vert ${SHADERS_SRC}/${PROJECT_NAME}_bg.frag ${SHADERS_SRC}/${PROJECT_NAME}_bg.vert)
	add_custom_target(${COMPILE_SHADER_TARGET} ALL DEPENDS SHADER_COMPILE SOURCES ${SHADER_SOURCES})
	add_dependencies (${PROJECT_NAME} ${COMPILE_SHADER_TARGET})
	
	set_target_properties(${PROJECT_NAME} PROPERTIES WORKING_DIRECTORY ${CURRENT_WORKING_DIR})
    set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
    if(WIN32)
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE VS_DEBUGGER_WORKING_DIRECTORY ${CURRENT_WORKING_DIR})
    endif(WIN32)
    if(APPLE)
        set_target_properties(${PROJECT_NAME} PROPERTIES XCODE_GENERATE_SCHEME TRUE XCODE_SCHEME_WORKING_DIRECTORY ${CURRENT_WORKING_DIR})
    endif(APPLE)
	target_link_libraries(${PROJECT_NAME} glfw glm)
endfunction(buildProject)

# Build all examples
function(buildProjects)
	foreach(PROJECT ${PROJECTS})
		buildProject(${PROJECT})
	endforeach(PROJECT)
endfunction(buildProjects)

set(PROJECTS
	draw_with_PBR
)

buildProjects()
