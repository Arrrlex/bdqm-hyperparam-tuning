#!/usr/bin/env bash

conda env create -f env_gpu.yml
pip install git+https://github.com/ulissigroup/amptorch.git@MCSH_paper1_lmdb
wget https://github.com/medford-group/bdqm-vip/raw/master/data/amptorch_data/oc20_3k_train.traj
