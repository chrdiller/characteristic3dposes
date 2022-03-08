#!/usr/bin/env bash

unzip grab__s1.zip
unzip grab__s2.zip
unzip grab__s3.zip
unzip grab__s4.zip
unzip grab__s5.zip
unzip grab__s6.zip
unzip grab__s7.zip
unzip grab__s8.zip
unzip grab__s9.zip
unzip grab__s10.zip

unzip tools__subject_meshes__male.zip
unzip tools__subject_meshes__female.zip

unzip models_smplx_v1_1.zip
mv models/smplx/ .
rm -r models
