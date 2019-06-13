#!/bin/bash

declare -A lrdict
lrdict=(["adam"]=0.0005 ["rmsprop"]=0.00018)

declare -A betadict
betadict=(["gan"]=0.5 ["wgan"]=0. ["wgan-gp"]=0.)

for mode in "wgan" "gan" "wgan-gp"
do
    for optim in "rmsprop" "adam"
    do
        folder=samples/${mode}_${optim}
        rm -rf $folder
        echo ""
        echo "start run" ${mode} ${optim}
        echo ""
        python main.py --cuda --mode ${mode} --optim ${optim} --b1 ${betadict[${mode}]} --lrG ${lrdict[${optim}]} --lrD ${lrdict[${optim}]}  --experiment $folder
    done
done

#rm -rf samples/wgan_rmsprop
#python main.py --cuda --mode wgan --optim adam --lrG 0.00018 --lrD 0.00018 --b1 0 --experiment samples/wgan_rmsprop