# Function for building single example
function(buildExample EXAMPLE_NAME)
	SET(EXAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME})
	message(STATUS "Generating project file for example in ${EXAMPLE_FOLDER}")
	# Main
	# file(GLOB SOURCE *.cpp ${BASE_HEADERS} ${EXAMPLE_FOLDER}/*.cpp)
	SET(MAIN_CPP ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.cpp)
	if(EXISTS ${EXAMPLE_FOLDER}/main.cpp)
		SET(MAIN_CPP ${EXAMPLE_FOLDER}/main.cpp)
	ENDIF()
	if(EXISTS ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.h)
		SET(MAIN_HEADER ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.h)
	ENDIF()
	if(WIN32)
		add_executable(${EXAMPLE_NAME} WIN32 ${MAIN_CPP})
		target_link_libraries(${EXAMPLE_NAME} ${Vulkan_LIBRARY} ${WINLIBS})
	else(WIN32)
		add_executable(${EXAMPLE_NAME} ${MAIN_CPP})
		target_link_libraries(${EXAMPLE_NAME})
	endif(WIN32)

	set (RESOURCES_DIR ${CMAKE_CURRENT_BINARY_DIR}/resources)
	set(SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}/shaders)
	
	# Compile shader
	add_custom_command(
    OUTPUT SHADER_COMPILE
	COMMAND ${CMAKE_COMMAND} -E make_directory ${RESOURCES_DIR}
    COMMAND glslc ARGS ${SHADERS_DIR}/${EXAMPLE_NAME}.frag -o ${RESOURCES_DIR}/frag.spv
    COMMAND glslc ARGS ${SHADERS_DIR}/${EXAMPLE_NAME}.vert -o ${RESOURCES_DIR}/vert.spv
	WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Shaders"
    VERBATIM
    )

	set(COMPILE_SHADER_TARGET ${EXAMPLE_NAME}_shader)
	set(SHADER_SOURCES ${SHADERS_DIR}/${EXAMPLE_NAME}.frag ${SHADERS_DIR}/${EXAMPLE_NAME}.vert)
	add_custom_target(${COMPILE_SHADER_TARGET} ALL DEPENDS SHADER_COMPILE SOURCES ${SHADER_SOURCES})
	add_dependencies (${EXAMPLE_NAME} ${COMPILE_SHADER_TARGET})
	
	set_target_properties(${EXAMPLE_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${RESOURCES_DIR})
	set_target_properties(${EXAMPLE_NAME} PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF LINK_FLAGS /SUBSYSTEM:CONSOLE)
	target_link_libraries(${EXAMPLE_NAME} glfw glm)
endfunction(buildExample)

# Build all examples
function(buildExamples)
	foreach(EXAMPLE ${EXAMPLES})
		buildExample(${EXAMPLE})
	endforeach(EXAMPLE)
endfunction(buildExamples)

set(EXAMPLES
	draw_the_triangle
)

buildExamples()
