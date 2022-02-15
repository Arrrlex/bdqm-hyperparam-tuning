#!/usr/bin/env bash

conda env create -f env_gpu.yml
pip install git+https://github.com/ulissigroup/amptorch.git@MCSH_paper1_lmdb
