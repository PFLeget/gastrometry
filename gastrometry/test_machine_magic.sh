#!/bin/bash

if [ "x"$1 != "x" ] ; then
  target="/"$1
fi
maxjob=35
for chip in $(seq 0 103) ; do
    while test $(jobs | wc -l) -gt $maxjob ; do
       sleep 10
    done
    schip=$(printf "%03d" $chip)  
    make ${schip}/${target} &
    sleep 1
done
