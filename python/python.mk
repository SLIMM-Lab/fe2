PYTHON ?= python3
PYTHON_EXECUTABLE := $(shell $(PYTHON) -c 'import sys; print(sys.executable)')
PYTHON_INCLUDE := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("include"))')
PYBIND_INCLUDE := $(shell $(PYTHON) -c 'import pybind11; print(pybind11.get_include())')
PYTHON_VERSION := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("LDVERSION") or sysconfig.get_python_version())')
PYTHON_LIBDIR := $(shell $(PYTHON) -c 'import sys,sysconfig,pathlib; p=pathlib.Path(sys.base_prefix)/"lib"; print(p if (p/sysconfig.get_config_var("LDLIBRARY")).exists() else sysconfig.get_config_var("LIBDIR"))')
PYTHON_EXTENSION := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
PYTHON_INCLUDES = -isystem $(PYTHON_INCLUDE) -isystem $(PYBIND_INCLUDE)
