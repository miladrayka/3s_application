3S_Application
--
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/miladrayka/3s_application.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/miladrayka/3s_application/context:python)

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)


Straightforwarding Scoring Suite (3S) is a collection of several tools to ease the procedure of desiging a machine learning scoring function by providing a GUI.

![](https://github.com/miladrayka/3s_application/blob/main/logo.png)

Contact
---
Milad Rayka, Chemistry and Chemical Engineering Research Center of Iran, milad.rayka@yahoo.com

Citation
--

1-[ET-score: Improving Protein-ligand Binding Affinity Prediction Based on Distance-weighted Interatomic Contact Features Using Extremely Randomized Trees Algorithm](https://onlinelibrary.wiley.com/doi/full/10.1002/minf.202060084)

2-[GB-Score: Minimally Designed Machine Learning Scoring Function Based on Distance-weighted Interatomic Contact Features](https://chemrxiv.org/engage/chemrxiv/article-details/6210b55ce0f5297c08b7f36a)

3-[Impact of non-normal error distributions on the benchmarking and ranking of quantum machine learning models](https://iopscience.iop.org/article/10.1088/2632-2153/aba184/meta)

Installation
--
Below packages should be installed for using 3S. Dependecies:

- python = 3.7

- numpy = 1.22.2

- pandas = 1.4.1

- seaborn = 0.11.2

- streamlit = 1.6.0

- matplotlib = 3.5.1

- biopandas = 0.3.0

- scipy = 1.8.0

- scikit-learn = 1.0.2

- progressbar2 = 4.0.0

- xgboost = 1.5.2

- notebook = 6.4.8

For installing first make a virtual environment and activate it.

On windows:

>    python py -m venv env

>    .\env\Scripts\activate

On macOS and Linux:

>    python3 -m venv env

>    source env/bin/activate

Which *env* is the location to create the virtual environment. Now you can install packages with one of the following methods:

>   1- pip install *package_name*

>  2 - python setup.py install

> 3 - python install -r requirements.txt

Usage
--

![](https://github.com/miladrayka/3s_application/blob/main/sample_gui.JPG)
*View of 3S GUI*


So far, this suite contains five tools:

1-Feature Generation:

In this mode, features for different structure of complexes based on aforementioned method are genereted[1].

2-Model Training:

In this mode, a machine learning scoring function (Gradient Boosting Trees) is designed for a dataset of provided complex structures.

3-Prediction:

Binding affinity of complexes are predicted using a ML-Score.

4-Normality Test:

In this mode, if the test data has binding label, normality property of errors is
analysed.

5-Add Hydrogen:

Add hydrogens to ligand and protein at pH=7.4 using PDB2PQR and Openbabel.

Check the provided **Tutorial.pdf** file for more information and example.

Development
--

To ensure code quality and consistency the following tools are used during development:

- [black](https://black.readthedocs.io/en/stable/)
- [isort](https://pycqa.github.io/isort/)
- [LGTM](https://lgtm.com/)

Copyright
--

Copyright (c) 2021-2022, Milad Rayka

