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

	set (CURRENT_WORKING_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	set_target_properties(${EXAMPLE_NAME} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if(WIN32)
        set_target_properties(${EXAMPLE_NAME} PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif(WIN32)
    if(APPLE)
        set_target_properties(${EXAMPLE_NAME} PROPERTIES XCODE_GENERATE_SCHEME TRUE XCODE_SCHEME_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
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
	empty_window
)

buildExamples()
