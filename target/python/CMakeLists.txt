FILE(GLOB_RECURSE SRC_FILES ../../src/*.c ../../src/*.cpp ../../src/*.cu ../../src/*.cc)
ADD_LIBRARY(${PROJECT_NAME}Python SHARED ${SRC_FILES})
set_target_properties(${PROJECT_NAME}Python PROPERTIES OUTPUT_NAME _blobflow)
install (TARGETS ${PROJECT_NAME}Python DESTINATION lib)