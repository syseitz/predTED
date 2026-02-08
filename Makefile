# Makefile for predTED
#
# Targets:
#   lib   — Build the Python C extension (via pip install -e .)
#   cli   — Build the standalone CLI binary (requires LightGBM)
#   test  — Run the test suite
#   clean — Remove build artefacts

UNAME_S := $(shell uname -s)

CC       ?= gcc
CXX      ?= c++
CFLAGS   ?= -O2 -Wall -Wno-deprecated-declarations
LDFLAGS  ?=

CLI_SRC   = c_src/predTED.c c_src/predted_features.c
CLI_BIN   = bin/predted

# --- Platform-specific LightGBM + OpenMP detection ---

ifeq ($(UNAME_S),Darwin)
  # macOS: find LightGBM from the Python package, OpenMP from Homebrew
  LGBM_LIB_DIR ?= $(shell python -c "import lightgbm, os; print(os.path.join(os.path.dirname(lightgbm.__file__), 'lib'))" 2>/dev/null)
  LGBM_INC     ?= $(shell \
    if [ -d "LightGBM/include" ]; then echo "-ILightGBM/include"; \
    elif [ -d "/opt/homebrew/include/LightGBM" ]; then echo "-I/opt/homebrew/include"; \
    fi)
  LGBM_LINK    ?= -L$(LGBM_LIB_DIR) -l_lightgbm -Wl,-rpath,$(LGBM_LIB_DIR)
  OMP_INC      ?= $(shell brew --prefix libomp 2>/dev/null)/include
  OMP_LIB      ?= $(shell brew --prefix libomp 2>/dev/null)/lib
  OMP_FLAGS     = -I$(OMP_INC) -Xpreprocessor -fopenmp -L$(OMP_LIB) -lomp
  # Use C++ compiler on macOS (LightGBM headers require C++)
  CLI_CC        = $(CXX)
else
  # Linux: use pkg-config or defaults
  LGBM_INC     ?= $(shell pkg-config --cflags-only-I LightGBM 2>/dev/null || echo "")
  LGBM_LINK    ?= $(shell pkg-config --libs LightGBM 2>/dev/null || echo "-lLightGBM")
  OMP_FLAGS     = -fopenmp
  CLI_CC        = $(CC)
endif

.PHONY: lib cli test clean

lib:
	pip install -e .

cli: $(CLI_BIN)

$(CLI_BIN): $(CLI_SRC) c_src/predted_features.h c_src/model.h
	@mkdir -p bin
	$(CLI_CC) $(CFLAGS) $(LGBM_INC) -Ic_src $(OMP_FLAGS) -o $@ $(CLI_SRC) $(LGBM_LINK) -lm $(LDFLAGS)

test:
	python -m pytest tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info predted/*.so predted/__pycache__
	rm -f $(CLI_BIN)
