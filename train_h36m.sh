#!/usr/bin/env bash

if [ ! -d "$(pwd)/experiments" ]; then mkdir "$(pwd)/experiments"; fi
if [ ! -d "$(pwd)/experiments/h36m" ]; then mkdir "$(pwd)/experiments/h36m"; fi

echo "Starting training for right hand"
if [ -d "$(pwd)/experiments/h36m/1" ]; then rm -r "$(pwd)/experiments/h36m/1"; fi
mkdir "$(pwd)/experiments/h36m/1"
env PYTHONPATH=. poetry run python characteristic3dposes/train.py hydra.run.dir="$(pwd)/experiments/h36m/1" data.type=h36m data.file="$(pwd)/data_generation/h36m/output/data_3d_h36m.npz" training.joint.predict=[16] training.joint.given=[] training.epochs=500 training.offsets.after_epoch=250

echo "Starting training for left hand"
if [ -d "$(pwd)/experiments/h36m/2" ]; then rm -r "$(pwd)/experiments/h36m/2"; fi
mkdir "$(pwd)/experiments/h36m/2"
env PYTHONPATH=. poetry run python characteristic3dposes/train.py hydra.run.dir="$(pwd)/experiments/h36m/2" data.type=h36m data.file="$(pwd)/data_generation/h36m/output/data_3d_h36m.npz" training.joint.predict=[13] training.joint.given=[16] training.epochs=500 training.offsets.after_epoch=250

echo "Starting training for the rest of the body"
if [ -d "$(pwd)/experiments/h36m/3" ]; then rm -r "$(pwd)/experiments/h36m/3"; fi
mkdir "$(pwd)/experiments/h36m/3"
env PYTHONPATH=. poetry run python characteristic3dposes/train.py hydra.run.dir="$(pwd)/experiments/h36m/3" data.type=h36m data.file="$(pwd)/data_generation/h36m/output/data_3d_h36m.npz" training.joint.predict=[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15] training.joint.given=[13,16] training.epochs=500 training.offsets.after_epoch=250
