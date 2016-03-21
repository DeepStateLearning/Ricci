

Observations:  
1) It works much better when we localize the Ricci flow - this makes perfect geometric sense. From now on we do this.  This does allow for possible sparse matrix computation later on. 

2) The algorithm works much better when we increase the value t a little bit.  I believe that in the past the t value was too small, so the points where almost behaving like they were discrete.  In any case, putting t = .05 up to t = .35 gave me the expected clustering for the noisy circles and in some cases the moons.  For t very small, the algorithm clusters noisy circles quite well when the factor is less than .5.   At .5 the picture seems to split almost symmetrically into quarters or thirds.  
3) I tried adding some far away noise, so that the rescaling wouldn't force things that should be clustered to split up for no reason on than the rescaling need to preserve distance.  This forced the noisy moons in the same cluster every time.  Perhaps there is more clever way to do the rescaling.  

4) I would like to convince myself at some point that this is not a Rube Goldberg machine that simply moves points that are close together more close together, and then rescale.  Since the Ricci curvature is often positive, this could be what's actually happen.  We'll have to try returning a Ricci curvature that is a copy of sqdist and see if this changes the results at all.  


