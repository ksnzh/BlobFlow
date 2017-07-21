FILE(GLOB_RECURSE SRC_FILES ../../src/*.c ../../src/*.cpp ../../src/*.cu ../../src/*.cc)
ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME blobflow)
install (TARGETS ${PROJECT_NAME} DESTINATION bin)