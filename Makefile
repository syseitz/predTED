# Makefile for predTED
#
# Targets:
#   lib   — Build the Python C extension (via pip install -e .)
#   cli   — Build the standalone CLI binary (requires LightGBM)
#   clean — Remove build artefacts

CC       ?= gcc
CFLAGS   ?= -O2 -Wall
LDFLAGS  ?=

# LightGBM (adjust if installed elsewhere)
LGBM_INC ?= $(shell pkg-config --cflags-only-I LightGBM 2>/dev/null || echo "")
LGBM_LIB ?= $(shell pkg-config --libs LightGBM 2>/dev/null || echo "-lLightGBM")

CLI_SRC   = c_src/predTED.c c_src/predted_features.c
CLI_BIN   = bin/predted

.PHONY: lib cli clean

lib:
	pip install -e .

cli: $(CLI_BIN)

$(CLI_BIN): $(CLI_SRC) c_src/predted_features.h c_src/model.h
	@mkdir -p bin
	$(CC) $(CFLAGS) $(LGBM_INC) -Ic_src -fopenmp -o $@ $(CLI_SRC) $(LGBM_LIB) -lm $(LDFLAGS)

clean:
	rm -rf build/ dist/ *.egg-info predted/*.so predted/__pycache__
	rm -f $(CLI_BIN)
