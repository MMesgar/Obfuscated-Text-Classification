#!/bin/bash


# run random
SEEDS=(0 1 2)
for SEED in ${SEEDS[@]};
   do
        python main.py --method random --seed $SEED > log-model-random-seed-$SEED.txt

   done


# run majority
SEEDS=(0 1 2)
for SEED in ${SEEDS[@]};
   do
        python main.py --method majority --seed $SEED > log-model-majority-seed-$SEED.txt
   done

METHOD=logreg
SEEDS=(0 1 2)
for SEED in ${SEEDS[@]};
   do
        python main.py --method $METHOD --seed $SEED > log-model-$METHOD-seed-$SEED.txt
   done


METHOD=mlp
SEEDS=(0 1 2)
for SEED in ${SEEDS[@]};
   do
        python main.py --method $METHOD --seed $SEED > log-model-$METHOD-seed-$SEED.txt
   done


METHOD=mlp
SEEDS=(0 1 2)
for SEED in ${SEEDS[@]};
   do
        python main.py --method $METHOD --seed $SEED > log-model-$METHOD-seed-$SEED.txt
   done

METHOD=cnn
SEEDS=(2)
for SEED in ${SEEDS[@]};
   do
        python main.py --method $METHOD --seed $SEED > log-model-$METHOD-seed-$SEED.txt
   done


METHOD=bilstm
SEEDS=(2)
for SEED in ${SEEDS[@]};
   do
        python main.py --method $METHOD --seed $SEED > log-model-$METHOD-seed-$SEED.txt
   done

METHOD=cnnbilstm
SEEDS=(1)
for SEED in ${SEEDS[@]};
   do
        python main.py --method $METHOD --seed $SEED > log-model-$METHOD-seed-$SEED.txt
   done