# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/u14014610/clion-2017.3.1/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/u14014610/clion-2017.3.1/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing/cmake-build-release

# Include any dependencies generated for this target.
include CMakeFiles/RGB_Processing.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/RGB_Processing.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RGB_Processing.dir/flags.make

CMakeFiles/RGB_Processing.dir/main.cpp.o: CMakeFiles/RGB_Processing.dir/flags.make
CMakeFiles/RGB_Processing.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/RGB_Processing.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RGB_Processing.dir/main.cpp.o -c /home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing/main.cpp

CMakeFiles/RGB_Processing.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RGB_Processing.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing/main.cpp > CMakeFiles/RGB_Processing.dir/main.cpp.i

CMakeFiles/RGB_Processing.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RGB_Processing.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing/main.cpp -o CMakeFiles/RGB_Processing.dir/main.cpp.s

CMakeFiles/RGB_Processing.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/RGB_Processing.dir/main.cpp.o.requires

CMakeFiles/RGB_Processing.dir/main.cpp.o.provides: CMakeFiles/RGB_Processing.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/RGB_Processing.dir/build.make CMakeFiles/RGB_Processing.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/RGB_Processing.dir/main.cpp.o.provides

CMakeFiles/RGB_Processing.dir/main.cpp.o.provides.build: CMakeFiles/RGB_Processing.dir/main.cpp.o


# Object files for target RGB_Processing
RGB_Processing_OBJECTS = \
"CMakeFiles/RGB_Processing.dir/main.cpp.o"

# External object files for target RGB_Processing
RGB_Processing_EXTERNAL_OBJECTS =

RGB_Processing: CMakeFiles/RGB_Processing.dir/main.cpp.o
RGB_Processing: CMakeFiles/RGB_Processing.dir/build.make
RGB_Processing: CMakeFiles/RGB_Processing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable RGB_Processing"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RGB_Processing.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RGB_Processing.dir/build: RGB_Processing

.PHONY : CMakeFiles/RGB_Processing.dir/build

CMakeFiles/RGB_Processing.dir/requires: CMakeFiles/RGB_Processing.dir/main.cpp.o.requires

.PHONY : CMakeFiles/RGB_Processing.dir/requires

CMakeFiles/RGB_Processing.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RGB_Processing.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RGB_Processing.dir/clean

CMakeFiles/RGB_Processing.dir/depend:
	cd /home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing /home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing /home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing/cmake-build-release /home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing/cmake-build-release /home/u14014610/Year3/Parallel-and-Concurrent-Programming/Coursework/RGB_Processing/cmake-build-release/CMakeFiles/RGB_Processing.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/RGB_Processing.dir/depend

