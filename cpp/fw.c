// metricize as ext_function for scipy.weave
const int n = Nd[0];

// sqdist to dist
#pragma omp parallel for schedule(static)
for (int i=0; i<n*n; i++)
{
    d[i] = sqrt(d[i]);
    //d2[i] = sqrt(d2[i]);
}

fw(d, n);
//fwC(d2, n);

// dist to sqdist
#pragma omp parallel for
for (int i=0; i<n*n; i++)
{
    d[i] *= d[i];
    //d2[i] *= d2[i];
}
