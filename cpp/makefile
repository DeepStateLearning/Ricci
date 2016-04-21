NUMCORE = $(shell python -c "import mkl; print mkl.get_max_threads()")
CFLAGS = -fopenmp -Wall -Werror -fPIC -fomit-frame-pointer -mfpmath=sse -O3 -DNUMCORE=$(NUMCORE)
# -fopt-info-vec 

all: libgenmul.so libgenmul_pureC.so libfw.so libgraph.so

libgraph.so: graph.cpp
	$(CXX) -o graph.o -c -Wno-sign-compare -march=native $(CFLAGS) graph.cpp
	$(CXX) -shared -o libgraph.so graph.o
	cp libgraph.so ../

libgenmul.so: genmul.c genmul.h genmul_defs.h macro.c metricize_dgemm.c
	$(CC) -o genmul.o -c -march=native $(CFLAGS) genmul.c
	$(CC) -o metricize.o -c -march=native $(CFLAGS) metricize_dgemm.c
	$(CC) -shared -o libgenmul.so genmul.o metricize.o $(LIBOMP)
	cp libgenmul.so ../

libgenmul_pureC.so: genmul.c genmul.h genmul_defs.h metricize_dgemm.c
	$(CC) -o genmul_pureC.o -c -ffast-math -march=native -fno-strict-aliasing -ftree-vectorize $(CFLAGS) genmul.c
	$(CC) -o metricize_pureC.o -c -ffast-math -march=native -fno-strict-aliasing -ftree-vectorize $(CFLAGS)  metricize_dgemm.c
	$(CC) -shared -o libgenmul_pureC.so genmul_pureC.o metricize_pureC.o $(LIBOMP)
	cp libgenmul_pureC.so ../

libgenmul_sse.so: genmul.c genmul.h genmul_defs.h
	$(CC) -o genmul_sse.o -c -msse3 $(CFLAGS) genmul.c
	$(CC) -shared -o libgenmul_sse.so genmul_sse.o $(LIBOMP)
	cp libgenmul_sse.so ../

libfw.so: floyd_warshall.c floyd_warshall.h macro.c
	$(CC) -o fw.o -c -march=native -ftree-vectorize $(CFLAGS) floyd_warshall.c 
	$(CC) -shared -o libfw.so fw.o $(LIBOMP)
	cp libfw.so ../

clean:
	rm -f *.so
	rm -f *.o
