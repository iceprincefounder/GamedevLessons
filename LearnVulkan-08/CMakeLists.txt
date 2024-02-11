# Function for building single example
include(CMakeParseArguments)

function(AddIconToBinary AppSources)
    set(Options)
    set(OneValueArgs OUTFILE_BASENAME)
    set(MultiValueArgs ICONS)
    cmake_parse_arguments(ARG "${Options}" "${OneValueArgs}" "${MultiValueArgs}" ${ARGN})
    if (NOT ARG_ICONS)
        message(FATAL_ERROR "No ICONS argument given to AddIconToBinary")
    endif()
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unexpected arguments to ecm_add_app_icon: ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    foreach (icon ${ARG_ICONS})
        get_filename_component(IconFull ${icon} ABSOLUTE)
        get_filename_component(IconType ${IconFull} EXT)
        get_filename_component(IconName ${IconFull} NAME_WE) 
        if (APPLE)
            if (${IconType} STREQUAL ".icns")
                set(IconFullOutput ${CMAKE_CURRENT_BINARY_DIR}/${IconName}.icns)
                configure_file(${IconFull} ${IconFullOutput} COPYONLY)
                set(MACOSX_BUNDLE_ICON_FILE ${IconName}.icns PARENT_SCOPE)
                set(${AppSources} "${${AppSources}};${IconFullOutput}" PARENT_SCOPE)
                set_source_files_properties(${IconFullOutput} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
                return()
            endif()            
        endif()
        if (MSVC)        
            if (${IconType} STREQUAL ".ico")
                set(IconFullOutput ${CMAKE_CURRENT_BINARY_DIR}/${IconName}.ico)
                configure_file(${IconFull} ${IconFullOutput} COPYONLY)
                file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${IconName}.rc.in" "IDI_ICON1 ICON DISCARDABLE\"${IconName}.ico\"\n")
                add_custom_command(
                        OUTPUT "${IconName}.rc"
                        COMMAND ${CMAKE_COMMAND}
                        ARGS -E copy "${IconName}.rc.in" "${IconName}.rc"
                        DEPENDS "${IconName}.ico"
                        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
                set(${AppSources} "${${AppSources}};${IconName}.rc" PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()
    return()
endfunction()

function(buildProject PROJECT_NAME)
	SET(EXAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME})
	message(STATUS "Generating project file for example in ${EXAMPLE_FOLDER}")
	# Main
	# file(GLOB SOURCE *.cpp ${BASE_HEADERS} ${EXAMPLE_FOLDER}/*.cpp)
	SET(MAIN_CPP ${EXAMPLE_FOLDER}/${PROJECT_NAME}.cpp)
    set (CURRENT_WORKING_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	set (ICONS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/appicons)
	set (SHADERS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/shaders)
	set (RESOURCES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/resources)
    set (RESOURCES_DEST ${CURRENT_WORKING_DIR}/Resources)
    set (SHADERS_DEST ${CURRENT_WORKING_DIR}/Resources/Shaders)
    set (ICONS_DEST ${CURRENT_WORKING_DIR}/Resources/Appicons)
    set (ASSETSETS_DEST ${CURRENT_WORKING_DIR}/Resources/Profabs)

	if(EXISTS ${EXAMPLE_FOLDER}/main.cpp)
		SET(MAIN_CPP ${EXAMPLE_FOLDER}/main.cpp)
	endif()
	if(EXISTS ${EXAMPLE_FOLDER}/${PROJECT_NAME}.h)
		SET(MAIN_HEADER ${EXAMPLE_FOLDER}/${PROJECT_NAME}.h)
	endif()
    # Set application icon
    AddIconToBinary(MAIN_CPP ICONS ${ICONS_SRC}/vulkan_renderer.ico ${ICONS_SRC}/vulkan_renderer.icns)
	if(WIN32)
        add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
        add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
		add_executable(${PROJECT_NAME} WIN32 ${MAIN_CPP})
		target_link_libraries(${PROJECT_NAME} ${Vulkan_LIBRARY} ${WINLIBS})
	else(WIN32)
		add_executable(${PROJECT_NAME} ${MAIN_CPP})
		target_link_libraries(${PROJECT_NAME})
	endif(WIN32)

	# Compile shader and copy texures
	add_custom_command(
    OUTPUT SHADER_COMPILE
	COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DEST}
    COMMAND glslc ARGS -g ${SHADERS_SRC}/${PROJECT_NAME}_base.frag -o ${SHADERS_DEST}/${PROJECT_NAME}_base_frag.spv
    COMMAND glslc ARGS -g ${SHADERS_SRC}/${PROJECT_NAME}_base.vert -o ${SHADERS_DEST}/${PROJECT_NAME}_base_vert.spv
    COMMAND glslc ARGS -g ${SHADERS_SRC}/${PROJECT_NAME}_base_instanced.vert -o ${SHADERS_DEST}/${PROJECT_NAME}_base_instanced_vert.spv
    COMMAND glslc ARGS -g ${SHADERS_SRC}/${PROJECT_NAME}_sm.frag -o ${SHADERS_DEST}/${PROJECT_NAME}_sm_frag.spv
    COMMAND glslc ARGS -g ${SHADERS_SRC}/${PROJECT_NAME}_sm.vert -o ${SHADERS_DEST}/${PROJECT_NAME}_sm_vert.spv
    COMMAND glslc ARGS -g ${SHADERS_SRC}/${PROJECT_NAME}_sm_instanced.vert -o ${SHADERS_DEST}/${PROJECT_NAME}_sm_instanced_vert.spv
    COMMAND glslc ARGS -g ${SHADERS_SRC}/${PROJECT_NAME}_bg.frag -o ${SHADERS_DEST}/${PROJECT_NAME}_bg_frag.spv
    COMMAND glslc ARGS -g ${SHADERS_SRC}/${PROJECT_NAME}_bg.vert -o ${SHADERS_DEST}/${PROJECT_NAME}_bg_vert.spv
    COMMAND glslc ARGS -g ${SHADERS_SRC}/${PROJECT_NAME}_sky.frag -o ${SHADERS_DEST}/${PROJECT_NAME}_sky_frag.spv
    COMMAND glslc ARGS -g ${SHADERS_SRC}/${PROJECT_NAME}_sky.vert -o ${SHADERS_DEST}/${PROJECT_NAME}_sky_vert.spv
	WORKING_DIRECTORY ${SHADERS_SRC}
    DEPENDS ${SHADERS_SRC} ${SHADER_SOURCES}
    COMMENT "Compiling Shaders Success!"
	VERBATIM
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${ICONS_SRC}/ ${ICONS_DEST}/
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCES_SRC}/textures/ ${RESOURCES_DEST}/Textures/
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCES_SRC}/models/ ${RESOURCES_DEST}/Models/
    COMMENT "Copying Resources Success!"
    VERBATIM
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSETSETS_DEST}
    )

	set(COMPILE_SHADER_TARGET ${PROJECT_NAME}_shader)
	set(SHADER_SOURCES ${SHADERS_SRC}/${PROJECT_NAME}_base.frag ${SHADERS_SRC}/${PROJECT_NAME}_base.vert ${SHADERS_SRC}/${PROJECT_NAME}_bg.frag ${SHADERS_SRC}/${PROJECT_NAME}_bg.vert ${SHADERS_SRC}/${PROJECT_NAME}_sm.frag ${SHADERS_SRC}/${PROJECT_NAME}_sm.vert)
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
	draw_with_instance
)

buildProjects()
