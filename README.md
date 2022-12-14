# Source code and data accompanying the article _Conditional particle filters with bridge backward sampling_

This repository contains the research source code and data behind the article 
[Karppinen, S., Singh S. S., Vihola, M. (2022) _Conditional particle filters with bridge backward sampling_](https://arxiv.org/abs/2205.13898)

## Installation 

1. Install [R](https://cran.r-project.org/mirrors.html) and run the following command in R:
```
install.packages(c("ggplot2", "dplyr", "patchwork", "latex2exp", "RColorBrewer"))
```
This will install all necessary R packages required by the code in this repository. 

2. Install [Julia version 1.8.2](https://julialang.org/downloads/) 
3. Clone this repository. (on the command line `git clone https://github.com/skarppinen/cpf-bbs.git`, or via GUI at GitHub)
4. Launch Julia in the root of the repository and execute the commands:
```
import Pkg;
Pkg.activate(".");
Pkg.instantiate();
```
The above downloads and installs all necessary Julia packages based on the Project.toml and Manifest.toml files.

## Obtaining input and simulation data

Input and simulation data are available for download from the following links: 

* [Input data](https://nextcloud.jyu.fi/index.php/s/d8WP6gGtyJaZDAM/download)
* [Simulation data](https://nextcloud.jyu.fi/index.php/s/TFGoKE6Ys9W56ts/download)

The input data (179 MB) contains data (CP-RBM and CTCRW-T datasets, Corine Land Cover, blocking sequences) required to run some the experiments (see below).
The simulation data (6.3 GB) is a compressed archive containing postprocessed simulation data that was used to draw the conclusions of the article. 

To download the input and simulation data, the convenience scripts `download-input-data.jl` and `download-simulation-data.jl` at the root folder of the
repository may be used.
To invoke them, simply call `julia *SCRIPTNAME*` on the command line at the root folder of the repository.
Using the download scripts has the additional benefit that the downloaded files will be extracted and placed to the correct folders assumed by the code in this repository. 

## Directory and file structure

The following documents the directory structure of this repository.
Note that in some file and function names "bbcpf" ("block backward conditional particle filter", earlier working name of the developed CPF) refers to the CPF-BBS, and is sometimes used instead of "cpfbbs", which is the abbreviation used in the article. For similar reasons, the CP-RBM model of the experiments section is sometimes referenced with "lgcpr" (short for "log-Gaussian Cox process, reflected").

* `src/julia/lib/` contains the source code of the methods developed in the article with the exception of conditional killing resampling and conditional systematic resampling with mean partition (Algorithms 5 and 6 in the article), which may be found from the separate package [Resamplings.jl](https://github.com/skarppinen/Resamplings.jl). Note that `Resamplings` will be automatically installed if the instructions above are followed. 

* `src/julia/models/` contains source code for the models appearing in the article. Note that the CTCRWP model in the article corresponds to the model named "CTCRWP_B", and the CTCRW-T model to the model named "CTCRWH". 

* `src/julia/scripts/` contains script files that setup and run the experiments of the paper (with adjustable settings). 
The scripts prefixed with `run-` run a full or partial experiment. Each script prefixed with `run-` features a help menu, which describes all parameters that the script accepts. To access said menu, call `julia *SCRIPTNAME* --help`. The following list shortly summarises each script:

	1. `run-bbcpf-blocksize-ctcrwp.jl` may be used to run the experiments with the CTCRWP model. 
	2. `run-bbcpf-lgcpr.jl` may be used to run experiments with the CP-RBM model. 
	3. `run-block-metrics-sys-ctcrwp.jl` computes blocking for CTCRWP model. 
	4. `run-block-metrics-sys-ctcrwt.jl` computes blocking for the experiment with CTCRW-T model.
	5. `run-block-metrics-sys-lgcpr.jl` computes blocking sequence selection data for the CP-RBM model.
	6. `run-lgcpr-determine-dyadic-blocking-block-metrics.jl` computes blocking for CP-RBM model based on blocking sequence selection data.
	7. `run-postprocess-lgcpr.jl` a postprocessing script for outputs of CP-RBM inference.
	8. `run-build-ctcrwt-data.jl` generates a dataset for the CTCRW-T experiment.
	9. `run-sim-lgcpr-data.jl` simulates data for the CP-RBM experiment.
	10. `run-terrain-sim.jl` runs the CTCRW-T experiment.
	11. `gen-task-lists.jl` generates the task list files in `src/bash/` (see below). 
	
	Additionally, `script-config.jl` contains settings for the scripts beginning with `run-`. Further note that data generated by some of the above scripts is already
	available in the input data discussed above.

* The Jupyter notebook `src/julia/notebooks/simulation-experiments.ipynb` contains the code for all figures of the article. 
The figures are visible in the notebook, and the notebook may be viewed directly on GitHub (without running any code, or cloning the project). 
Some of the figures differ slightly from those of the article because of random number generation, but the code and conclusions are identical to
the article. Note that running the code in the notebook requires that the simulation data has been downloaded as discussed above.

* `src/bash/` contains so called tasklists that document the parameters with which the experiments of the
scripts folder (see above) were run in order to produce the results of the article. The environment variable `BBCPF_JL_SCRIPT_FOLDER`
is used to determine the location of the scripts. Note that fully executing the tasklists is computationally intensive and in practice requires
a computing cluster. However, the downloadable simulation data discussed above contains the relevant data generated by executing the tasklists.

* `src/R/r-plot-helpers.R` is an R file containing some functionality and settings for drawing figures (in the notebook above).


