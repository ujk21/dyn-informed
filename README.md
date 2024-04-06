# Dynamics-informed protein design with structure conditioning
This repository contains a minimal example to reproduce adenylate kinase (3adk) example from https://openreview.net/forum?id=jZPqf2G9Sw.

## Download & setup

*Conda environment and Genie submodule*
Create a conda environment for the project and download the Genie submodule
```bash
conda env create -f requirements/env_gpu.yml
conda activate dinf-env
git submodule init
git submodule update
cd genie
pip install -e .
cd ..
pip install -e .
```

### Example run
To run a simple example, using joint conditioning with reconstruction guidance for substructure and NMA dynamics conditioning, run
```bash
# activate env with the required dependencies
conda activate dinf-env

# run conditional sampling
python scripts/genie_cond_sampling.py --structure_on --dynamics_on
```

## Overview

- Unconditional Genie model is downloaded as a submodule (`genie`)
- The structure conditioner and the dynamics conditioner are in `src/diffusion/structconditioner.py` and  `src/diffusion/eigenconditioner.py`respectively. Both can be modified to use a problem-specific time scaling of the guidance scales.
- To take samples with the 3adk hinge target, run `scipts/genie_cond_sampling.py`
