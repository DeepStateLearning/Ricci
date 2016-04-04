// depth-first connected components
int n = Nd[0];

std::vector<std::vector<int> > components(n);

// Each vertex is with itself first
for (unsigned i=0;i<n;i++)
    components[i].push_back(i);

for (unsigned i=0;i<n;i++)
{
    // go through rows of the matrix
    // find cluster for vertex i
    //
    // if already visited then skip
    if (components[i][0]<0) continue;
    unsigned j=0;
    do
    {
        // visited -> skip
        int v = components[i][j];
        components[i][0] = -i-1;
        // not yet analysed row with index v
        // find all connections for v 
        for (unsigned k=0;k<n;k++)
        {
            // no connection or visited -> skip
            if (d[v*n+k]>threshold || components[k][0]<0) continue;
            // k is the new element of the component 
            components[i].push_back(k);
            // mark row k as cluster (-i-1)
            components[k][0]=-i-1;
        }   
     }
     while (++j<components[i].size());
}

// fill colors array with cluster indices
int color = 0;
for (unsigned i=0;i<n;i++)
{
    if (components[i].size()>1 || components[i][0]==-i-1)
    {
        // std::cout<<i<<" ";
        colors[i] = color;
        for (unsigned j=1;j<components[i].size();j++)
        {
            // std::cout<<components[i][j]<<" ";
            colors[components[i][j]] = color;
        }
        // std::cout<<std::endl<<std::endl;
        color++;
    }
}

return_val = color;
/*
for (unsigned i=0; i<n; i++)
    std::cout<<colors[i]<<" ";
std::cout<<std::endl;
*/

