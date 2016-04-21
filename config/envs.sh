CONDABIN=`find ~ /opt -path "*conda/bin" -type d -print -quit`

if [[ $OSTYPE == darwin* ]]; then 
    echo "Setting up for Mac with gcc-5"
    gcc-5 -v 2> /dev/null || (echo "!! Error !!" && exit 1)
    export PATH="$CONDABIN:$PATH"
    export CXX="g++-5"
    export CC="gcc-5"
    export LIBRARY_PATH="/usr/local/lib:$LIBRARY_PATH"
    export CPATH=":/usr/local/include:$CPATH"
    export OMP_DYNAMIC=FALSE
    export OMP_PROC_BIND=TRUE
    export LIBOMP="-lgomp"
else
    echo "Setting up for Linux with clang-3.8"
    clang-3.8 -v 2> /dev/null || (echo "!! Error !!" && exit 1)
    CONDALIB=`find ~ /opt -path "*conda/lib" -type d -print -quit`
    export PATH="$CONDABIN:$PATH"
    export LIBRARY_PATH=".:$CONDALIB"
    export LD_LIBRARY_PATH=".:$CONDALIB"
    export CPATH=`echo '#include <omp.h>' | cpp -H -o /dev/null 2>&1 | head -n1 | cut -d\  -f2 | rev | cut -d/ -f2- | rev`
    export OMP_DYNAMIC=FALSE
    export OMP_PROC_BIND=TRUE
    export CC=clang-3.8
    export CXX=clang++-3.8
    export LIBOMP="-liomp5"
fi
