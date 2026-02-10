#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# TVAC Thermal Analyzer v4.0 - Build Script
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║  TVAC Thermal Analyzer v4 – Build        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Step 1: Check Python dependencies ──────────────────────────
echo "[1/3] Checking Python dependencies..."
MISSING=0

python3 -c "import PyQt5" 2>/dev/null || {
    echo "  ✗ PyQt5 not found"
    MISSING=1
}
python3 -c "import numpy" 2>/dev/null || {
    echo "  ✗ numpy not found"
    MISSING=1
}
python3 -c "import scipy" 2>/dev/null || {
    echo "  ✗ scipy not found"
    MISSING=1
}

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "  Installing missing dependencies..."
    pip3 install PyQt5 numpy scipy 2>/dev/null || \
    pip install PyQt5 numpy scipy 2>/dev/null || {
        echo "  ✗ Failed to install. Please run manually:"
        echo "    pip install PyQt5 numpy scipy"
        exit 1
    }
fi
echo "  ✓ All Python dependencies available"

# ── Step 2: Compile C thermal engine ───────────────────────────
echo ""
echo "[2/3] Compiling C thermal engine..."

CC="${CC:-gcc}"
CFLAGS="-O3 -march=native -shared -fPIC -Wall -Wno-stringop-overflow"
SRC="thermal_engine.c"

# Detect platform
case "$(uname -s)" in
    Linux*)   OUT="libthermal_engine.so";;
    Darwin*)  OUT="libthermal_engine.dylib"; CFLAGS="$CFLAGS -dynamiclib";;
    MINGW*|MSYS*|CYGWIN*) OUT="thermal_engine.dll";;
    *)        OUT="libthermal_engine.so";;
esac

# Check for OpenMP support
HAS_OPENMP=0
echo "int main(){return 0;}" > /tmp/_omp_test.c
if $CC -fopenmp /tmp/_omp_test.c -o /tmp/_omp_test 2>/dev/null; then
    HAS_OPENMP=1
    CFLAGS="$CFLAGS -fopenmp"
    echo "  ✓ OpenMP support detected"
else
    echo "  ⚠ OpenMP not available (single-threaded build)"
fi
rm -f /tmp/_omp_test.c /tmp/_omp_test

if [ ! -f "$SRC" ]; then
    echo "  ✗ $SRC not found!"; exit 1
fi

$CC $CFLAGS -o "$OUT" "$SRC" -lm 2>&1 && {
    echo "  ✓ Compiled: $OUT ($(du -h "$OUT" | cut -f1))"
} || {
    echo "  ⚠ C compilation failed. Will use Python/SciPy fallback solver."
    echo "    (This is OK but ~3-5x slower)"
}

# ── Step 3: Verify ─────────────────────────────────────────────
echo ""
echo "[3/3] Verifying installation..."

python3 -c "
import sys
sys.path.insert(0, '.')
# Verify syntax
import py_compile
py_compile.compile('tvac_analyzer.py', doraise=True)
print('  ✓ Python code: OK')

# Verify C engine
try:
    import ctypes
    lib = ctypes.CDLL('./$OUT')
    ver = lib.thermal_get_version
    ver.restype = ctypes.c_char_p
    print(f'  ✓ C engine: loaded (v{ver().decode()})')
except:
    print('  ⚠ C engine: not available (using SciPy fallback)')

# Verify PyQt5
try:
    from PyQt5.QtWidgets import QApplication
    print('  ✓ PyQt5: OK')
except:
    print('  ✗ PyQt5: MISSING - GUI will not work!')
    sys.exit(1)
"

echo ""
echo "═══════════════════════════════════════════"
echo " Build complete!"
echo ""
echo " Run with:"
echo "   python3 tvac_analyzer.py"
echo "   python3 tvac_analyzer.py board.kicad_pcb"
echo "═══════════════════════════════════════════"
