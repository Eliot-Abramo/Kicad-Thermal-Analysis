#!/bin/bash
# Build script for TVAC Thermal Analyzer native engine
# 
# This compiles the C thermal simulation engine with OpenMP support
# for significant performance improvements (10-100x faster than Python)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==================================="
echo "TVAC Thermal Engine Build Script"
echo "==================================="
echo ""

# Detect platform
PLATFORM=$(uname -s)
echo "Platform: $PLATFORM"

# Set output name based on platform
if [ "$PLATFORM" = "Darwin" ]; then
    OUTPUT="libthermal_engine.dylib"
    SHARED_FLAG="-dynamiclib"
elif [ "$PLATFORM" = "Linux" ]; then
    OUTPUT="libthermal_engine.so"
    SHARED_FLAG="-shared"
else
    echo "Warning: Unknown platform, assuming Linux-like"
    OUTPUT="libthermal_engine.so"
    SHARED_FLAG="-shared"
fi

# Check for compiler
if command -v gcc &> /dev/null; then
    CC=gcc
elif command -v clang &> /dev/null; then
    CC=clang
else
    echo "Error: No C compiler found (gcc or clang required)"
    exit 1
fi

echo "Compiler: $CC"

# Check for OpenMP support
echo ""
echo "Checking OpenMP support..."
OMP_FLAG=""
if $CC -fopenmp -E - < /dev/null &> /dev/null; then
    OMP_FLAG="-fopenmp"
    echo "  OpenMP: enabled"
else
    echo "  OpenMP: not available (single-threaded build)"
fi

# Build options
CFLAGS="-O3 -fPIC -Wall"
LDFLAGS="-lm"

# Add architecture-specific optimizations
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    CFLAGS="$CFLAGS -march=native -mtune=native"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    CFLAGS="$CFLAGS -mcpu=native"
fi

echo ""
echo "Build configuration:"
echo "  Output: $OUTPUT"
echo "  CFLAGS: $CFLAGS $OMP_FLAG"
echo "  LDFLAGS: $LDFLAGS"
echo ""

# Compile
echo "Compiling thermal_engine.c..."
$CC $CFLAGS $OMP_FLAG $SHARED_FLAG -o "$OUTPUT" thermal_engine.c $LDFLAGS

if [ -f "$OUTPUT" ]; then
    echo ""
    echo "Build successful!"
    echo "Output: $SCRIPT_DIR/$OUTPUT"
    
    # Show file info
    ls -lh "$OUTPUT"
    
    # Show symbols (optional)
    if command -v nm &> /dev/null; then
        echo ""
        echo "Exported symbols:"
        nm -gU "$OUTPUT" 2>/dev/null | grep -E "^[0-9a-f]+ T " | head -20 || true
    fi
    
    echo ""
    echo "The native engine will be automatically loaded by the plugin."
else
    echo ""
    echo "Build failed!"
    exit 1
fi
