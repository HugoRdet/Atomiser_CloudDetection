#!/bin/bash
source /etc/profile.d/lmod.sh
module load conda
conda activate venv

python3 test_toto.py
