// only tiny bit faster than numexpr
// only works for even n
const size_t n = Nd[0]*Nd[1];

#define buf_size 10000
static double buffer[buf_size]  __attribute__ ((aligned (16)));
const size_t chunks = n / buf_size; 

#pragma omp parallel for schedule(static) private(buffer)
for (size_t i=0; i<chunks; i++)
{
    for(size_t j=0; j<buf_size; j++)
        buffer[j] = d[i*buf_size+j];
    for (size_t j=0; j<buf_size; j+=2)
    {
        __m128d pd = _mm_load_pd(&buffer[j]);
        pd = _mm_sqrt_pd(pd);
        _mm_store_pd(&buffer[j], pd);
    }
    for(size_t j=0; j<buf_size; j++)
        d[i*buf_size+j] = buffer[j];
}
            
