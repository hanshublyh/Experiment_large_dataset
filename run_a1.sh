#!/bin/bash
a="80_200"
b="100_300"
c="120_400"
d="150_500"
exp="NoExp"
r="r3"
l="r1+r2"
for file in a b c d;do
    cd $file
    for i in 1 2 3 4 5 6 7 8 9 10; do
        cd $i
        cd $exp
        cd $r
        python a1.py >> a1.txt
        cd ..
        cd $l 
        python a1.py >> a1.txt
        cd ../../../
    done
    cd ..
done