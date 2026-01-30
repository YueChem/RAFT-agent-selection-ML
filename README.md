Data-driven Multi-objective Optimization Strategy for RAFT Agent Selection in Copolymerization – Supporting Information Data

This repository provides the Supporting Information (SI) data and scripts for the manuscript:

Data-driven Multi-objective Optimization Strategy for RAFT agent selection in copolymerization

The repository contains the complete workflows for molecular descriptor extraction, database construction, machine learning model development, and model validation used for RAFT agent screening.



1. Descriptors (1-descriptors/)

This directory contains scripts and related files for molecular descriptor extraction of both monomers and RAFT agents.

The descriptors are generated from the corresponding molecular structures and are used as the input features for all machine learning models developed in this work.


2. Database (2-database/)

This directory contains the complete molecular structure database used in this study, including:

all monomer structures, and

all RAFT agent structures

considered in the data-driven screening and optimization workflow.


3. Machine learning and validation (3-machine-learning/)

This directory contains all scripts, data, and trained models related to machine learning model development and validation, including:

descriptor preprocessing (data cleaning, normalization and feature scaling),

feature analysis and feature selection,

hyperparameter optimization and model training,

multi-objective learning and stacking strategies,

trained optimal machine learning models saved as .pkl files, and

internal cross-validation and external validation workflows, including performance evaluation and statistical analysis.


Summary of provided contents

This repository provides:

descriptor extraction workflows for both monomers and RAFT agents,

the complete molecular structure database used in this study,

full machine learning pipelines including descriptor preprocessing, feature analysis, and hyperparameter optimization,

trained optimal models (.pkl), and

all scripts required to reproduce the model training and validation results reported in the manuscript.


Citation

If you use the data or codes in this repository, please cite the following manuscript:

Data-driven Multi-objective Optimization Strategy for RAFT agent selection in copolymerization
