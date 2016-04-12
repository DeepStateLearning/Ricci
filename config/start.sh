#!/bin/bash
xhost +
socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\" &
IP=`ifconfig vboxnet0 | grep "inet " | cut -d\  -f2`
docker run -e "DISPLAY=$IP:0" -v /Users/siudeja/Dropbox/Software/other/ricci:/home/ricci -it ricci /bin/bash -c "cd /home/ricci; /bin/bash"
killall socat
xhost -
