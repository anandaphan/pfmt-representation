# Piano Fingering Representation in Evolutionary Multitasking Optimization
This repository contains the code supplementary to the paper:

[**Diagnosing Optimizer-Landscape Interaction via Representation in Subset Selection Multitasking: Evidence from Piano Fingering**](https://doi.org/10.1109/CEC69861.2026.11727126)

Paper : [link](https://doi.org/10.1109/CEC69861.2026.11727126)

If you use this code in whole or part of your research, please cite our paper:
```text
@inproceedings{iman2026diagnosing,
  author    = {Iman, Ananda Phan and Wei, Tingyang and Ahn, Chang Wook},
  title     = {Diagnosing Optimizer-Landscape Interaction via Representation in Subset Selection Multitasking: Evidence from Piano Fingering},
  booktitle = {2026 IEEE Congress on Evolutionary Computation (CEC)},
  year      = {2026},
  pages     = {1--8},
  keywords  = {Multitasking;Optimization;Fingering;Algorithms;Knowledge transfer;Autocorrelation;Encoding;Music;Decoding;Art and music;Multi-task optimization;Fitness landscape analysis;Continuous relaxation;Subset selection},
  doi       = {10.1109/CEC69861.2026.11727126}
}
```


## Overview
This work investigates how different continuous representations reformulate the piano fingering estimation problem within a unified search space for evolutionary multitasking optimization, and how these representations affect optimization behavior through the induced fitness landscape.

By treating piano fingering as a representative subset selection multitasking problem, we examine how alternative encodings reshape the geometry of the search space and influence both solution quality and cross-task knowledge transfer. This allows us to study, in a controlled setting, the interaction between representation design, landscape characteristics, and optimizer behavior, including how properties such as ruggedness and neutrality relate to multitasking performance.


## Representation Studies

| Rep ID | Representation Type | Mapping | Problem Name in MATLAB |
|--------|----------------------|---------|-------------------------|
| R1 | Direct Representation | Partition-based pointwise mapping | R1-PF-MTSO[1-8] |
| R2 | Modular Homogeneous Representation | Top-$k_j$ truncation mapping | R2-PF-MTSO[1-8] |
| R3 | Modular Heterogeneous Representation | Sequential selection without replacement | R3-PF-MTSO[1-8] |
| R4 | Indirect Representation | Fixed VAE decoder, p = 2 | R4-PF-MTSO[1-8] |
| R5 | Indirect Representation | Fixed VAE decoder, p = 1 | R5-PF-MTSO[1-8] |

## 1. Environment Prerequisite
To reproduce the paper, make sure you have the following installed:

```text
Python Environment
- python==3.10.0
- music21==9.1.0
- pandas==2.12
- numpy==1.26.0
- torch==2.2.2
- matplotlib==3.8.0

MATLAB
- MATLAB >= R2023b
- Statistics and Machine Learning Toolbox
- pyenv configured in MATLAB and connected to the Python environment
```
Alternatively, follow this [Documentation](https://www.mathworks.com/help/matlab/matlab_external/install-supported-python-implementation.html) to connect Matlab to an existing Python environment


## 2. Usage
### Main Experiment

Pieces come from the PIG dataset. Its license does not allow redistribution, so you must download it yourself (free registration)

Therefore to run the experiment, follow these steps:
1. Download the fingering dataset from [PIG Dataset](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/) and place the .txt files inside each folder at "[Rep ID]/Tasks/PIG",
2. Clone the [MTO-Platform (MToP)](https://github.com/intLyc/MTO-Platform) 
3. Copy each representation folder from "representation/[Rep ID]" into MToP directory  "MTO/Problems/PianoFingering/"
4. Run the MTO Platform in MATLAB using the selected algorithm

Note: To avoid file conflict, experiments from different representations should be run separately. For example, do not run R1 together with R2 or R3. However, all cases within a single representation can be executed simultaneously.

In the case conflicts arise, run the following command in Matlab to restart the Python Environment Setting inside matlab or restart the matlab program entirely.:
```
terminate(pyenv)
clear all
clc
```

### Export .mat into .csv
To export the matlab.mat file to csv:
1. Place "script/export.mat" in the MToP folder "MTO-Platform/MTO/"
2. Run the following command in the MATLAB terminal
```
matlab -batch "export_csvs('path-to-matlab-file.mat', 'path-to-export-folder dir')"
``` 
3. The result will be explored in the structurex
```
<path-to-export-folder dir>/<Case_Number>/<Algorithm_name>/<Run_number>.csv
``` 
