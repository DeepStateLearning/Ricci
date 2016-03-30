bool clustered_depthfirst(unsigned n, double threshold, const double *d)
{
    // Check symmetry and diagonal
    /*
    for (unsigned i=0; i<n; i++)
    {
        if (abs(d[i*n+i]) > 1e-10) return 0;
        for (unsigned j=i+1; j<n; j++)
            if (abs(d[i*n+j]-d[j*n+i])>1e-10) return 0;
    }
    */

    std::vector<std::vector<int> > components(n);

    // Each vertex is with itself first
    for (unsigned i=0;i<n;i++)
        components[i].push_back(i);

    for (unsigned i=0;i<n;i++)
    {
        // skip visited rows
        if (components[i][0]<0) continue;
        // mark row i as cluster (-i-1)
        components[i][0] = -i-1;
        // find all connections to i (only above index i needed)
        for (unsigned j=i+1;j<n;j++)
        {
            // no connection -> skip
            if (d[i*n+j]>threshold) continue;
            // visited means not clustered (connection to earlier cluster)
            if (components[j][0]<0) return false;
            // j is the new element of the cluster 
            components[i].push_back(j);
            // mark row j as cluster (-i-1)
            components[j][0]=-i-1;
        }   
        // We should have all vertices from the cluster.
        // No new connections should exist in rows belonging to the cluster.
        for (unsigned j=1; j<components[i].size(); j++)
        {
            unsigned row = components[i][j]; // current row
            // only check for connections above index row
            // anything below will clash when the next cluster runs into this marked connection
            // E.g. 1<->4<->2, but not 1<->2, then 4 in here will miss the check against 2
            //      but 4 will already be marked when row 2 is analysed
            for (unsigned k=row+1; k<n; k++)
            {
                if (d[row*n+k]<=threshold && components[k][0]!=-i-1) return false; // new or wrong connection
                if (d[row*n+k]>threshold && components[k][0]==-i-1) return false; // missing connection
            }
        }
    }
    /*
    for (unsigned i=0;i<n;i++)
    {
        if (components[i].size()>1 || components[i][0]==-i-1)
        {
            std::cout<<i<<" ";
            for (unsigned j=1;j<components[i].size();j++)
                std::cout<<components[i][j]<<" ";
            std::cout<<std::endl<<std::endl;
        }
    }
    */
    return true;
}
