



1) It works much better when we localize the Ricci flow - this makes perfect geometric sense. From now on we do this.  This does allow for possible sparse matrix computation later on. 

2) The algorithm works much better when we increase the value t a little bit.  I believe that in the past the t value was too small, so the points where almost behaving like they were discrete.  In any case, putting t = .05 up to t = .35 gave me the expected clustering for the noisy circles and in some cases the moons.  For t very small, the algorithm clusters noisy circles quite well when the factor is less than .5.   At .5 the picture seems to split almost symmetrically into quarters or thirds.  


3) I tried adding some far away noise, so that the rescaling wouldn't force things that should be clustered to split up for no reason on than the rescaling need to preserve distance.  This forced the noisy moons in the same cluster every time.  Perhaps there is more clever way to do the rescaling.  

4) The flow is currently unstable - this is probably not good.  Creating a complete graph, and then adding a very small amount of noise, and running the ricci flow does end up clustering some points with each other.  Ideally, it should recreate a complete graph which is perfect homogeneous.    My thoughts at this point is that the volume needs to be rescaled to really reflect how the Ricci flow is behaving.  



Observations:  
0) I'm not sure this whole thing isn't a Rube Goldberg machine that is no more than modifying the distance function so that nearby points become closer, remetricizing, and then rescaling.  I just added a dumbricci test that simply return the square distance (i.e Ricci = Id) and it appears to get better results in some cases.   It's not clear right away that the Ricci is actually helping.   


update:  Ricci seems to be working much betterm faster and more frequently than dumb Ricci

update: update:  Confirmed many times again.  Dumb does get lucky, but in most of the cases it either fails to converge or converges quite slowly.  The ricci curvature flow is working much much better than the dumb one. 
