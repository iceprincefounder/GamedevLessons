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
	set (TEXTURE_DIR ${CURRENT_WORKING_DIR}/Textures)
	set (SHADERS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/shaders)
	set (TEXTURE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/textures)

	# Compile shader and copy texures
	add_custom_command(
    OUTPUT SHADER_COMPILE
	COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    COMMAND glslc ARGS ${SHADERS_SRC}/${PROJECT_NAME}.frag -o ${SHADERS_DIR}/${PROJECT_NAME}_frag.spv
    COMMAND glslc ARGS ${SHADERS_SRC}/${PROJECT_NAME}.vert -o ${SHADERS_DIR}/${PROJECT_NAME}_vert.spv
	WORKING_DIRECTORY ${SHADERS_SRC}
    DEPENDS ${SHADERS_SRC} ${SHADER_SOURCES}
	VERBATIM
	COMMENT "Compiling Shaders!"
	COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURE_SRC}/botw_mifa.png ${TEXTURE_DIR}/botw_mifa.png
	COMMAND ${CMAKE_COMMAND} -E copy ${TEXTURE_SRC}/background.png ${TEXTURE_DIR}/background.png
	COMMENT "Copying Textures!"
    )

	set(COMPILE_SHADER_TARGET ${PROJECT_NAME}_shader)
	set(SHADER_SOURCES ${SHADERS_SRC}/${PROJECT_NAME}.frag ${SHADERS_SRC}/${PROJECT_NAME}.vert)
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
	draw_the_scene
)

buildProjects()
