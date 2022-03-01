Example folder contains 3 subfolders:

1 - files:

In this folder, all files for running example are provided. 

`x_set.csv` : Generated features based on pdbbind 2019 (general + refined - core sets).

`y_set.csv`: Labels of `x_set.csv`.

`x_test_set.csv`: Generated features for the pdbbind 2016 core set.

`y_test_set.csv`: Labels of `x_test_set.csv`.

`test_set_pdbid.csv`: PDBids of the core set.

`prediction_test_set.csv`: CSV file contains GB-Score prediction and label of the core set.

`columns_pdbbind_2019.txt`: Text file contains retained features names after preprocessing (Use for prediction by GB-Score).

`mean_pdbbind_2019.csv`: CSV file contains mean of retained features after preprocessing (Use for prediction by GB-Score).

`std_pdbbind_2019.csv`: CSV file contains std of retained features after preprocessing (Use for prediction by GB-Score).

2 - structures:

In this folder, ten structures (*.mol2* for ligand and *.pdb* for protein) are provided for testing feature generation and adding hydrogens.

3 - model:

In this folder, *GB-Score* trained model is located. This model can be used for prediction task. For more information check [2]. 