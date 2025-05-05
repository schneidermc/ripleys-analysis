# Ripley's K Analysis

This repository contains code for performing Ripley’s K analysis, a spatial point pattern analysis technique commonly used in microscopy and spatial statistics.

The core functionality of this package has been incorporated into the [`picasso-workflow`](https://github.com/jungmannlab/picasso-workflow) repository maintained by the Jungmann lab at the Max Planck Institute of Biochemistry. You can find it as a submodule at [`picasso_workflow/ripleys_analysis`](https://github.com/jungmannlab/picasso-workflow/tree/master/picasso_workflow/ripleys_analysis).

### Setup
To run this package individually, run the following line from the terminal to create and activate a new environment `ripleys`:
```bash
conda env create -f environment.yaml
conda activate ripleys
```

### Run analysis
To perform Ripley’s K analysis, set the filenames for your data in `run_ripleysAnalysis.py`, then run:
```bash
python run_ripleysAnalysis.py
```
