#!/usr/bin/env bash

pace-jupyter-notebook \
  -q pace-ice-gpu \
  -l nodes=1:ppn=1:gpus=1 \
  -l walltime=3:00:00 \
  --conda-env=bdqm-hpopt \
  --lab
