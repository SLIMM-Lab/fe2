.DEFAULT_GOAL := extension
include python.mk
program = material_bridge
OBJDIR = OBJ_EXTENSION
extension = _fe2_material_native$(PYTHON_EXTENSION)
VPATH = ../src
objects = $(filter-out main.cpp PRNNMaterial.cpp,$(notdir $(wildcard ../src/*.cpp)))
MY_CXX_STD_FLAGS = -std=c++17 -fPIC -Wno-unused-local-typedefs -Werror -DFE2_PYTHON_EXTENSION $(PYTHON_INCLUDES)
MY_LIBDIRS = $(PYTHON_LIBDIR)
MY_LIBS = python$(PYTHON_VERSION)
MY_CXX_LNK_FLAGS = -shared -Wl,-rpath,$(PYTHON_LIBDIR)
include $(JEMDIR)/packages/mt/Makedefs.mk
include $(JIVEDIR)/makefiles/prog.mk
include $(JIVEDIR)/makefiles/packages/*.mk

.PHONY: extension
extension: $(extension)
$(extension): $(objects)
	$(LinkExe)
