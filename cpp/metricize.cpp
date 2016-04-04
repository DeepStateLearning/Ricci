// metricize as ext_function for scipy.weave
double error=1;
const int n = Nd[0];
// reshape lower triangle (r entries) into a rectangle with width w
const int r = n*(n-1)/2;
const int w = n%2 ? (n-1)/2 : (n-1);
std::vector<double> tril;
tril.resize(r);

// sqdist to dist
#pragma omp parallel for schedule(static)
for (int i=0; i<n*n; i++)
    d[i] = sqrt(d[i]);

while (error > 10e-12)
{
    error = 0;
#pragma omp parallel shared(error)
    {
        double d_ij, dijk, old;
#pragma omp for reduction(max:error) schedule(static, w)
        for (int l=0; l<r; ++l)
        {
            // lower triangle from linear index to improve parallel loop
            int i = l/w;
            int j = l%w;
            if (j>=i)
            {
                i = n-1-i;
                j = n-2-j;
            }
            old = d_ij = d[i*n+j];
            for (int k=0; k<n; ++k)
            {
                dijk = d[i*n+k] + d[j*n+k];
                if (dijk < d_ij)
                    d_ij = dijk;
            }
            if (old>d_ij)
            {
                if (error < old-d_ij)
                    error = old-d_ij;
                tril[l] = d_ij;
            }
        }
#pragma omp for schedule(static, w)
        for (int l=0; l<r; ++l)
            if (tril[l]>0)
            {
                // lower triangle from linear index to improve parallel loop
                int i = l/w;
                int j = l%w;
                if (j>=i)
                {
                    i = n-1-i;
                    j = n-2-j;
                }
                d[i*n+j] = d[j*n+i] = tril[l];
                tril[l] = 0;
            }
    } // parallel
} // while

// dist to sqdist
#pragma omp parallel for
for (int i=0; i<n*n; i++)
    d[i] *= d[i];
