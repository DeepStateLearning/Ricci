

Observations:  
1) It works much better when we localize the Ricci flow - this makes perfect geometric sense. From now on we do this.  This does allow for possible sparse matrix computation later on. 

2) The algorithm clusters noisy circles quite well when the factor is less than .5.   At .5 the picture seems to split almost symmetrically into quarters or thirds.  
3) I tried adding some far away noise, so that the rescaling wouldn't force things that should be clustered to split up for no reason on than the rescaling need to preserve distance.  This forced the noisy moons in the same cluster every time.  Perhaps there is more clever way to do the rescaling.  


