#! /bin/bash
# make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=4 -r 16 -t 56 --seed 44 --reg=frob,1e-9,1,2,3,4 ~/hpctensor/sm_flickr.tns
#make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=4 -r 16 -t 56 --seed 44 --reg=frob,1e-12,1,2,3,4 ~/hpctensor/sm_flickr.tns
# make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=4 -r 16 -t 56 --seed 44 --reg=frob,1e-12,1,2,3,4 ~/hpctensor/sm_flickr.tns
#make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=4 -r 16 -t 56 --seed 44 --reg=frob,1e-12,1,2,3,4 ~/hpctensor/sm_flickr.tns
# make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=1 -r 16 -t 56 --seed 44 --reg=frob,1e-12,1,2,3,4 ~/hpctensor/uber.tns



#uber
#make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=1 -r 16 -t 56 --seed 44 --reg=frob,1e-12,1,2,3,4 ~/hpctensor/uber.tns
#make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=1 -r 16 -t 56 --seed 44 ~/hpctensor/uber.tns

#flickr
#make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=4 -r 16 -t 56 --seed 44 --reg=frob,1e-12,1,2,3,4 ~/hpctensor/flickr-4d.tns
#make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=4 -r 16 -t 56 --seed 44 --reg=frob,1e-12,1,2,3,4 --use_rsp=True ~/hpctensor/flickr-4d.tns

#make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=4 -r 16 -t 56 --seed 44 --reg=frob,1e-12,1,2,3,4 ~/hpctensor/uber.tns --con=nonneg
#make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=1 -r 16 -t 56 --seed 44 --use_rsp=true --reg=frob,1e-12,1,2,3,4 --use_rsp=True ~/hpctensor/uber.tns
make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=1 -r 16 -t 56 --seed 45 --use_rsp=true --reg=frob,1e-12,1,2,3,4 --use_rsp=True ~/hpctensor/uber.tns
#make && ./build/Linux-x86_64/bin/splatt cpd -v --stream=1 -r 4 -t 56 --seed 44 --use_rsp=true --reg=frob,1e-12,1,2,3,4 --use_rsp=True ~/hpctensor/uber.tns
