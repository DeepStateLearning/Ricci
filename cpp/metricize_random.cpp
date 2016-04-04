// metricize with somewhat random outcome due to 
// side-effects of updating the distance matrix in the loops
const int n = Nd[0];
double error = 1;

while (error>1e-12)
{
    error = 0;
//#pragma omp parallel for reduction(+:error) schedule(dynamic)
    for(int i=0; i<n; i++)
        for (int j=i+1; j<n; j++)
        {
            double old_dij, dij, dijk;
            old_dij = dij = d[i*n+j];
            for (int k=0; k<n; k++)
            {
                dijk = d[i*n+k] + d[j*n+k];
                dij = dijk < dij ? dijk : dij;
            }
            if (old_dij>dij)
            {
                d[i*n+j] = d[i+j*n] = dij;
                if (old_dij-dij>error)
                    error = old_dij-dij;
            }           
        }
    std::cout<<error<<" ";
}
std::cout<<std::endl;
