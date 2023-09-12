# Function for building single example
function(buildExample EXAMPLE_NAME)
	SET(EXAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME})
	message(STATUS "Generating project file for example in ${EXAMPLE_FOLDER}")
	# Main
	# file(GLOB SOURCE *.cpp ${BASE_HEADERS} ${EXAMPLE_FOLDER}/*.cpp)
	SET(MAIN_CPP ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.cpp)
	if(EXISTS ${EXAMPLE_FOLDER}/main.cpp)
		SET(MAIN_CPP ${EXAMPLE_FOLDER}/main.cpp)
	endif()
	if(EXISTS ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.h)
		SET(MAIN_HEADER ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.h)
	endif()
	if(WIN32)
		add_executable(${EXAMPLE_NAME} WIN32 ${MAIN_CPP})
		target_link_libraries(${EXAMPLE_NAME} ${Vulkan_LIBRARY} ${WINLIBS})
	else(WIN32)
		add_executable(${EXAMPLE_NAME} ${MAIN_CPP})
		target_link_libraries(${EXAMPLE_NAME})
	endif(WIN32)

    set (CURRENT_WORKING_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	set(SHADERS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}/shaders)
	set (SHADER_DEST ${CURRENT_WORKING_DIR}/Resources/Shaders)
	
	# Compile shader
	add_custom_command(
    OUTPUT SHADER_COMPILE
	COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_DEST}
    COMMAND glslc ARGS ${SHADERS_SRC}/${EXAMPLE_NAME}.frag -o ${SHADER_DEST}/${EXAMPLE_NAME}_frag.spv
    COMMAND glslc ARGS ${SHADERS_SRC}/${EXAMPLE_NAME}.vert -o ${SHADER_DEST}/${EXAMPLE_NAME}_vert.spv
	WORKING_DIRECTORY ${SHADERS_SRC}
    DEPENDS ${SHADERS_SRC} ${SHADER_SOURCES}
    COMMENT "Compiling Shaders"
    VERBATIM
    )

	set(COMPILE_SHADER_TARGET ${EXAMPLE_NAME}_shader)
	set(SHADER_SOURCES ${SHADERS_SRC}/${EXAMPLE_NAME}.frag ${SHADERS_SRC}/${EXAMPLE_NAME}.vert)
	add_custom_target(${COMPILE_SHADER_TARGET} ALL DEPENDS SHADER_COMPILE SOURCES ${SHADER_SOURCES})
	add_dependencies (${EXAMPLE_NAME} ${COMPILE_SHADER_TARGET})
	
	set_target_properties(${EXAMPLE_NAME} PROPERTIES WORKING_DIRECTORY ${CURRENT_WORKING_DIR})
    set_target_properties(${EXAMPLE_NAME} PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
    if(WIN32)
        set_target_properties(${EXAMPLE_NAME} PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE VS_DEBUGGER_WORKING_DIRECTORY ${CURRENT_WORKING_DIR})
    endif(WIN32)
    if(APPLE)
        set_target_properties(${EXAMPLE_NAME} PROPERTIES XCODE_GENERATE_SCHEME TRUE XCODE_SCHEME_WORKING_DIRECTORY ${CURRENT_WORKING_DIR})
    endif(APPLE)
	target_link_libraries(${EXAMPLE_NAME} glfw glm)
endfunction(buildExample)

# Build all examples
function(buildExamples)
	foreach(EXAMPLE ${EXAMPLES})
		buildExample(${EXAMPLE})
	endforeach(EXAMPLE)
endfunction(buildExamples)

set(EXAMPLES
	draw_with_buffers
)

buildExamples()
