#!/bin/bash

for filename in /export/c11/sli136/silver-dataset/translations/*/*.phrase/*.trans; do
    # echo `wc -l $filename`
    num=`wc -l $filename  | sed 's/ / /' | awk '{print $1}'`
    if [ $(($num % 100)) -ne 0 ]; then
        echo "$num $filename"
    fi
    if [ $num == 0 ]; then
        echo "$num $filename"
    fi
done