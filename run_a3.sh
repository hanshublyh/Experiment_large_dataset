#!/bin/bash
a="device.txt"
b="main.ipynb"
c="run_a1.sh"
d="run_a3.sh"
e="5_40"
f="readme.txt"
exp="NoExp"
r="r3"
l="r1+r2"
g=".git"
for file in `ls`;do
    if [ $file != $a -a $file != $b -a $file != $c -a $file != $d -a $file != $e -a $file != $f -a $file != $g ]
    then
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
    fi
done