program = fe2 

PYTHON_BRIDGE ?= 0
ifeq ($(HEADLESS),1)
HEADLESS_FLAGS = -DFE2_HEADLESS
endif
OBJDIR ?= OBJ$(if $(filter 1,$(HEADLESS)),_HEADLESS)$(if $(filter 1,$(PYTHON_BRIDGE)),_PYTHON)

ifeq ($(PYTHON_BRIDGE),1)
include ../python/python.mk
PYTHON_FLAGS = -DFE2_WITH_PYTHON -DFE2_PYTHON_EXECUTABLE='"$(PYTHON_EXECUTABLE)"' -DFE2_PYTHON_SOURCE='"$(abspath ../python)"' $(PYTHON_INCLUDES)
MY_LIBDIRS = $(PYTHON_LIBDIR)
MY_LIBS = python$(PYTHON_VERSION)
MY_CXX_LNK_FLAGS = -Wl,-rpath,$(PYTHON_LIBDIR)
else
objects = $(filter-out PRNNMaterial.cpp PythonMaterialBridge.cpp PythonMaterialBindings.cpp,$(wildcard *.cpp))
endif

include $(JEMDIR)/packages/mt/Makedefs.mk
include $(JIVEDIR)/makefiles/prog.mk
include $(JIVEDIR)/makefiles/packages/*.mk

MY_CXX_STD_FLAGS = -std=c++17 -Wno-unused-local-typedefs -Werror $(PYTHON_FLAGS) $(HEADLESS_FLAGS)

# Public executable name is shared by build modes, so always relink it.
.PHONY: $(program) $(program-opt) $(program-dbg)
