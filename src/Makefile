program = fe2 

CXX_STANDARD ?= c++17

# Numerical runs need no OpenGL modules; useful with a headless Jem/Jive build.
ifeq ($(HEADLESS),1)
HEADLESS_FLAGS = -DFE2_HEADLESS
endif

# Avoid reusing objects with an incompatible material registry or link flags.
OBJDIR ?= OBJ$(if $(filter 1,$(HEADLESS)),_HEADLESS)$(if $(strip $(LIBTORCH)),_PRNN)

# Opt in with LIBTORCH=/path/to/libtorch (or the Python torch package root).
# Set TORCH_CXX11_ABI to match the chosen distribution and Jem/Jive builds.
ifneq ($(strip $(LIBTORCH)),)
TORCH_CXX11_ABI ?= 1
# Treat third-party headers as system headers; keep -Werror for our sources.
PRNN_FLAGS = -DFE2_WITH_LIBTORCH -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_CXX11_ABI) -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include
MY_LIBDIRS = $(LIBTORCH)/lib
MY_LIBS = torch torch_cpu c10
MY_CXX_LNK_FLAGS = -Wl,-rpath,$(LIBTORCH)/lib
else
objects = $(filter-out PRNNMaterial.cpp,$(wildcard *.cpp))
endif

include $(JEMDIR)/packages/mt/Makedefs.mk
include $(JIVEDIR)/makefiles/prog.mk
include $(JIVEDIR)/makefiles/packages/*.mk

MY_CXX_STD_FLAGS = -std=$(CXX_STANDARD) -Wno-unused-local-typedefs -Werror $(PRNN_FLAGS) $(HEADLESS_FLAGS)

# Modes share the public executable name; relink it with the selected object set
# even when objects from another mode are newer than this mode's cached objects.
.PHONY: $(program) $(program-opt) $(program-dbg)
