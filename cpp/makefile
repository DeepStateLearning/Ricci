NUMCORE = $(shell python -c "import mkl; print mkl.get_max_threads()")
CFLAGS = -fopenmp -Wall -Werror -fpic -fomit-frame-pointer -mfpmath=sse -O3 -DNUMCORE=$(NUMCORE)

all: libgenmul.so libgenmul_pureC.so libgenmul_sse.so libfw.so

libgenmul.so: genmul.c genmul.h genmul_defs.h macro.c
	$(CC) -o genmul.o -c -march=native $(CFLAGS) genmul.c -lgomp
	$(CC) -shared -o libgenmul.so genmul.o -lgomp
	cp libgenmul.so ../

libgenmul_pureC.so: genmul.c genmul.h genmul_defs.h
	$(CC) -o genmul_pureC.o -c -ffast-math -march=native -ftree-vectorize $(CFLAGS) genmul.c -lgomp
	$(CC) -shared -o libgenmul_pureC.so genmul_pureC.o -lgomp
	cp libgenmul_pureC.so ../

libgenmul_sse.so: genmul.c genmul.h genmul_defs.h
	$(CC) -o genmul_sse.o -c -msse3 $(CFLAGS) genmul.c -lgomp
	$(CC) -shared -o libgenmul_sse.so genmul_sse.o -lgomp
	cp libgenmul_sse.so ../

libfw.so: floyd_warshall.c floyd_warshall.h macro.c
	$(CC) -o fw.o -c -march=native $(CFLAGS) floyd_warshall.c -lgomp
	$(CC) -shared -o libfw.so fw.o -lgomp
	cp libfw.so ../

clean:
	rm -f *.so
	rm -f *.o
