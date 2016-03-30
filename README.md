# Ricci

There are two goals of this project.   One is to explore if the notion of coarse Ricci curvature, as we've defined it, is effective in clustering.   A second goal is to explore if coarse Ricci curvature is useful for finding either uniform or otherwise somehow 'nice' metrics.  We expect these two things may be related.  

We are attempting to create a coarse Ricci flow by simply modifying the distance squared function by the coarse Ricci curvature times a small \eta at each step.  

For example, the hope is that a cluster of points near each other in some metric space, and a cluster of points far away, will have distance that becomes smaller within in each cluster and stays the same or grows between the two clusters.   There's some issues choosing rescaling but this seems to happen in some examples.  

Also, one would hope that if one perturbs a 'round' metric, the flow would behave in the long run like that of a round metric.  

This is based on our recent work,  where we define the coarse Ricci curvature in terms of iterated laplace type operators.  http://arxiv.org/abs/1505.04166. 

UPDATE:  It seems we need to play with densities in order to solve "finding the Einstein metric" part of the problem, but the Ricci flow is working to cluster some basic examples.   

![Image of 3d cluster ](https://raw.githubusercontent.com/micah541/Ricci/master/pictures/3d2.png)

![Image of Noisy Circles ](https://raw.githubusercontent.com/micah541/Ricci/master/pictures/NoisyCirclesMarch29.png)


![Image of Noisy Moons ](https://raw.githubusercontent.com/micah541/Ricci/master/pictures/NoisyMoonsMarch25.png)
