

Possible solution for Mac when compiling package from source:
1. install clang 4.0
brew install --with-clang llvm

2. adjust/create ~/.R/Makevars file and point R to use it
# put your paths to clang-4.0 and clang++-4.0:
CC=/usr/local/opt/llvm/bin/clang
CXX=/usr/local/opt/llvm/bin/clang++
CXX11=/usr/local/opt/llvm/bin/clang++
CXX1X=/usr/local/opt/llvm/bin/clang++
#
#And add runtime and headers
LDFLAGS    = -L/usr/local/opt/llvm/lib
CPPFLAGS   = -I/usr/local/opt/llvm/include
#
3. Optionally add CPU-specific optimizations flags (below my setup for modern core i7 with avx2):
CXX11FLAGS += -O3 -ffast-math -march=native -mavx2
CXXFLAGS   += -O3 -ffast-math -march=native -mavx2
CFLAGS     += -O3 -ffast-math -march=native -mavx2