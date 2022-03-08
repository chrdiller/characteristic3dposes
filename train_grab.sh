#!/usr/bin/env bash

if [ ! -d "$(pwd)/experiments" ]; then mkdir "$(pwd)/experiments"; fi
if [ ! -d "$(pwd)/experiments/grab" ]; then mkdir "$(pwd)/experiments/grab"; fi

echo "Starting training for right hand"
if [ -d "$(pwd)/experiments/grab/1" ]; then rm -r "$(pwd)/experiments/grab/1"; fi
mkdir "$(pwd)/experiments/grab/1"
env PYTHONPATH=. poetry run python characteristic3dposes/train.py hydra.run.dir="$(pwd)/experiments/grab/1" data.type=grab data.file="$(pwd)/data_generation/grab/output/data_3d_grab.npz" training.joint.predict=[4] training.joint.given=[] training.epochs=200 training.offsets.after_epoch=100

echo "Starting training for left hand"
if [ -d "$(pwd)/experiments/grab/2" ]; then rm -r "$(pwd)/experiments/grab/2"; fi
mkdir "$(pwd)/experiments/grab/2"
env PYTHONPATH=. poetry run python characteristic3dposes/train.py hydra.run.dir="$(pwd)/experiments/grab/2" data.type=grab data.file="$(pwd)/data_generation/grab/output/data_3d_grab.npz" training.joint.predict=[7] training.joint.given=[4] training.epochs=200 training.offsets.after_epoch=100

echo "Starting training for the rest of the body"
if [ -d "$(pwd)/experiments/grab/3" ]; then rm -r "$(pwd)/experiments/grab/3"; fi
mkdir "$(pwd)/experiments/grab/3"
env PYTHONPATH=. poetry run python characteristic3dposes/train.py hydra.run.dir="$(pwd)/experiments/grab/3" data.type=grab data.file="$(pwd)/data_generation/grab/output/data_3d_grab.npz" training.joint.predict=[0,1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] training.joint.given=[4,7] training.epochs=200 training.offsets.after_epoch=100
