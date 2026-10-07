program = prnn_material_test
subdirs = ../src
VPATH = ../src
objects = prnn_material_test.cpp PRNNMaterial.cpp J2Material.cpp IsotropicMaterial.cpp Material.cpp MaterialFactory.cpp FEMUtils.cpp Invariants.cpp voigtUtilities.cpp utilities.cpp

ifndef LIBTORCH
$(error Set LIBTORCH to a CPU LibTorch distribution or Python torch package root)
endif
CXX_STANDARD ?= c++17
TORCH_CXX11_ABI ?= 1
MY_INCDIRS = ../src
MY_LIBDIRS = $(LIBTORCH)/lib
MY_LIBS = torch torch_cpu c10
MY_CXX_LNK_FLAGS = -Wl,-rpath,$(LIBTORCH)/lib
MY_CXX_STD_FLAGS = -std=$(CXX_STANDARD) -Wno-unused-local-typedefs -Werror -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_CXX11_ABI) -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include

include $(JIVEDIR)/makefiles/prog.mk
include $(JIVEDIR)/makefiles/packages/*.mk
