int n = Nd[0];
int i;
int count=1;
double error;

error = 1;
while (error > 10e-12)
{
    error = 0;
    // shortest paths to d2
    dgemm_nn(n, d, d2);

#pragma omp parallel for private(i) shared(d, d2) reduction(max:error)
    for (i=0;i<n*n; i++)
    {
        if (error<d[i]-d2[i])
            error=d[i]-d2[i];
        d[i] = d2[i];
    }
  
    if ((limit > 0) && (++count > limit)) break;
    //printf("%f ", error);
}
