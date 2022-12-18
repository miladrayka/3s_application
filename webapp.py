import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from scipy.stats import kurtosis, probplot, shapiro, skew

from generate_features import generate_features
from predict import predict
from train import train_pipline

sys.tracebacklimit = 0

st.title("3*S* Web Application")

st.sidebar.header("Straightforwarding Scoring Suite")
st.sidebar.write(
    """**S**traightforwarding **S**coring **S**uite (3*S*) is a collection of several tools for devising new scoring 
    functions based on distance-weighted interatomic contact features and Gradient Boosting Trees algorithm for 
    the training procedure. 
    """
)
st.sidebar.subheader("Citation")

st.sidebar.write(
    """[1 - ET-score: Improving Protein-ligand Binding Affinity Prediction Based on 
Distance-weighted Interatomic Contact Features Using Extremely Randomized Trees 
Algorithm](https://onlinelibrary.wiley.com/doi/abs/10.1002/minf.202060084)\n
[2 - GB-Score: Minimally Designed Machine Learning Scoring Function Based on 
Distance-weighted Interatomic Contact Features](https://chemrxiv.org/engage/chemrxiv/article-details/6210b55ce0f5297c08b7f36a)\n
[3 - Impact of non-normal error distributions on the benchmarking and ranking of quantum machine 
learning models](https://iopscience.iop.org/article/10.1088/2632-2153/aba184/meta)
"""
)


st.sidebar.subheader("Mode")
add_selectbox = st.sidebar.selectbox(
    "Operation",
    (
        "Introduction",
        "1-Feature Generation",
        "2-Model Training",
        "3-Prediction",
        "4-Normality Test",
        "5-Add Hydrogen",
    ),
)


if add_selectbox == "Introduction":

    image = Image.open("logo.png",)

    st.image(image)

    st.write(
        """Straightforwarding Scoring Suite (3*S*) is a collection of several tools to ease the procedure
of desiging a machine learning scoring function. These tools are designed based on our recent papers which we
introduced a new scheme of feature generation based on distance-weighted interatomic contact and
using Gradient Boosting Trees as a machine learning algorithm.\n
So far, this suite contains five tools (accessible from the side panel):\n
1-Feature Generation\n
In this mode, features for different structure of complexes based on aforementioned method are genereted.\n
2-Model Training\n
In this mode, a machine learning scoring function (Gradient Boosting Trees) is designed for a dataset of 
provided complex structures.\n
3-Prediction\n
In this mode, a trained model is used for predicting binding score for unseen data.\n
4-Normality Test\n
In this mode, if the test data has binding label, normality property of errors is analysed.\n
5-Add Hydrogen\n
Adding hydrogen to ligand and protein structures.
"""
    )

if add_selectbox == "1-Feature Generation":

    st.subheader("Feature Generation")

    with st.expander("Theoretical Information"):

        st.write(
            """Element-based atom types 
{*H, C, N, O, F, P, S, Cl, Br, I*} are considered for ligand. 
To generate protein atom types, amino acid 
residues are classified based on their chemical nature of side-chains into 
four groups (*Charged* (**c**), *Polar* (**p**), *Amphipathic* (**a**), *Hydrophobic* (**h**)):\n
**Charged** = {*Arg, Lys, Asp, Glu*}\n
**Polar** = {*Gln, Asn, His, Ser, Thr, Cys*}\n
**Amphipathic** = {*Trp, Tyr, Met*}\n
**Hydrophobic** = {*Ile, Leu, Phe, Val, Pro, Gly, Ala*}\n
In the next step, all interatomic distances for a specific atom types pair are calculated. 
Distances with magnitude below the predefined cutoff  (**$d_{cutoff}$**)  are 
weighted by an inverse power of a natural number (**n**) and sum together:\n
"""
        )
        st.latex(
            r"\vec{X} = \left \{X_{H, H_{p}}, X_{H, C_{p}},...,X_{I, I_{h}}\right \}"
        )
        st.latex(r"X_{i,j} = \sum_{k=1}^{K_{j}}\sum_{l=1}^{L_{i}}\frac{1}{d_{lk}^{n}}")
        st.write(
            """where *i* and *j* are atom types of ligand and protein, respectively; 
$L_{i}$ is the total number of ligand atoms of type *i* and $K_{j}$ is the total number of 
protein atoms of type *j*, $d_{lk}$ is the Euclidean distance between the *l*-th 
ligand atom of type *i* and the *k*-th protein atom of type *j*, which is less than **$d_{cutoff}$**[1]."""
        )

    with st.expander("Caution about directory"):

        st.warning(
            "**Caution**: Directory of structures should be like the following picture:"
        )

        image2 = Image.open("tree.png")

        st.image(image2)

    with st.expander("Caution about structures"):

        st.warning(
            "**Caution**: Both ligand and protein structures should be hydrogenated. Use **Add Hyrogen** mode."
        )

    directory = st.text_input(
        "Enter directory of your complex structures:",
        value="Example/structures",
        help="Indicating path of complex structures. e.g /Example/structures",
    )

    assert os.path.isdir(directory), "Enter valid directory."

    path = Path(directory)

    for folder in path.iterdir():

        if not all(
            file.endswith(".mol2") or file.endswith(".pdb")
            for file in os.listdir(folder)
        ):

            st.error(
                """**Error**: Please correct your provided dicrectory. It contains files which their extension is not .mol2 
            or .pdb."""
            )

    exponent = st.number_input(
        "Enter exponent of weighting function:",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="Required exponent (*n*) for weighting features.",
    )

    st.info("Optimized value for *n* is 2.")

    cutoff = st.number_input(
        "Enter distance cutoff:",
        min_value=4.0,
        max_value=18.0,
        value=12.0,
        step=0.5,
        help="Required distance cutoff  (**$d_{cutoff}$**) for feature generation.",
    )

    st.info("Prefered value for (**$d_{cutoff}$**) is 12 A.")

    complexes = st.checkbox(
        "Single pdb file for complex.",
        value=False,
        help="""If your ligand and protein structures are 
                        in a single pdb (not seperate .mol2 for ligand 
                        and .pdb for protein) click on checkbox.""",
    )

    filename = st.text_input(
        "Enter output filename in .csv:",
        value="sample.csv",
        help="""Enter your desired filename 
                         with csv extension as an output file. e.g output.csv""",
    )

    assert (
        os.path.splitext(filename)[1][1:].strip().lower() == "csv"
    ), "Filename extension should be csv."

    execute = st.button(
        "Start feature generations",
        help="Press the button to start feature generations.",
    )

    if execute:

        start = time.time()

        with st.spinner("In progress..."):

            generate_features(directory, exponent, cutoff, complexes, filename)

        end = time.time()

        seconds = end - start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        st.success(f"Procedure is completed at: {h:.2f} h, {m:.2f} min, {s:.2f} sec!")


if add_selectbox == "2-Model Training":

    st.subheader("Model Training")

    with st.expander("Theoretical Information"):

        st.write(
            """An *[Extreme Gradient Boosting Trees (XGBT)](https://xgboost.readthedocs.io/en/stable/)* 
             is trained based on
             generated features in **Feature Generation** mode.
             Before training, a preprocess step is applied onto data.
             In this step, features with low variance (static and quasi-static) and 
             correlated ones are discarded and the rest are normalized using their mean and std."""
        )

    path_x = st.text_input(
        "Enter path of features data (.csv):",
        value="Example/files/x_set.csv",
        help="Indicating path of features data. e.g Example/files/x_set.csv",
    )

    assert os.path.exists(path_x), "Error: File doesn't exist."
    assert (
        os.path.splitext(path_x)[1][1:].strip().lower() == "csv"
    ), "File extension should be csv."

    st.warning(
        """**Caution**: Your target data csv file should has two columns with the following names: 
             **pdbid** and **binding_affinity**."""
    )

    path_y = st.text_input(
        "Enter path of target data (.csv):",
        value="Example/files/y_set.csv",
        help="Indicating path of target data. Example/files/y_set.csv",
    )

    assert os.path.exists(path_y), "Error: File doesn't exist."
    assert (
        os.path.splitext(path_y)[1][1:].strip().lower() == "csv"
    ), "File extension should be csv."

    st.warning(
        """**Caution**: Your test set csv file should has two columns with the following names: 
             **pdbid** and **binding_affinity**."""
    )

    path_test_id = st.text_input(
        "Enter path of test set pdbid (.csv):",
        value="Example/files/test_set_pdbid.csv",
        help="Indicating path of test set pdbid. e.g Example/files/test_set_pdbid.csv",
    )

    assert os.path.exists(path_test_id), "Error: File doesn't exist."
    assert (
        os.path.splitext(path_test_id)[1][1:].strip().lower() == "csv"
    ), "File extension should be csv."

    var_threshold = st.number_input(
        "Enter variance threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        step=0.01,
        help="Filter static and quasi-static features.",
    )

    corr_threshold = st.number_input(
        "Enter correlation threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01,
        help="Discard correlated features.",
    )

    # val_set_size = st.number_input(
    #    "Enter the percentage of validation set:",
    #    min_value=0.0,
    #    max_value=1.0,
    #    value=0.0,
    #    step=0.01,
    #    help="Create a validation set from the train set.",
    # )

    gpu = st.checkbox(
        "Use GPU accelerator during training.",
        value=False,
        help="XGBT uses GPU to accelerate training procedure.",
    )

    filename = st.text_input(
        "Enter output filename in .json:",
        value="Example/model/gb_score_cpu.json",
        help="""Enter your desired filename 
                         with joblib extension as an output file
                         to save the trained model. e.g trained_model.json""",
    )

    assert (
        os.path.splitext(filename)[1][1:].strip().lower() == "json"
    ), "File extension should be json."

    execute = st.button(
        "Start training operation", help="Press the button to start training.",
    )

    if execute:

        start = time.time()

        with st.spinner("In progress..."):

            train_pipline(
                path_x,
                path_y,
                path_test_id,
                var_threshold,
                corr_threshold,
                0,
                gpu,
                filename,
            )

        end = time.time()

        seconds = end - start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        st.success(f"Procedure is completed at: {h:.2f} h, {m:.2f} min, {s:.2f} sec!")


if add_selectbox == "4-Normality Test":

    st.subheader("Investigating Normality of an Error Distribution")

    sns.set_style("darkgrid")

    with st.expander("Theoretical Information"):
        st.write(
            """If the distribution of residual errors for a benchmark dataset is zero-centered normal, statistics such as
the mean absolute error (*MAE*), the root mean squared deviation (*RMSD*) or the $95^{th}$ quantile of the
absolute errors distribution ($Q_{95}$) are redundant and can be used to infer *u(M)*:
*u(M)* ≃ *RMSD* = √π /2*MAE* ≃ 0 .5$Q_{95}$. If it is not normal, more information is necessary to provide the
user with probabilistic diagnostics.\n
Statistical benchmarking of a method *M* is based on the estimation of errors 
(*EM* = {$e_{M, i}$| *i*=1,...,*N*}) for a set of *N* calculated (*CM* = {$C_{M, i}$| *i*=1,...,*N*}) and reference data 
(*R* = {$r_{i}$| *i*=1,...,*N*}), where:\n
$e_{M, i} = r_{i} - C_{M,i}$"""
        )

    st.warning(
        """**Caution**: Your csv file should has two columns with the following names: 
             **Target** and **Predicted**."""
    )

    try:
        uploaded_file = st.text_input(
            "Load your data (.csv)",
            value="Example/files/prediction_test_set.csv",
            help="Enter .csv file of your data. e.g Example/files/prediction_test_set.csv",
        )

        df = pd.read_csv(uploaded_file)

        st.dataframe(df)

        residual_error = df["Target"] - df["Predicted"]

        mse = np.mean(residual_error)
        rmsd = np.sqrt(np.mean((residual_error) ** 2))
        mae = np.mean(np.abs(residual_error))

        st.info(f"Mean of error: {mse:.3f}")
        st.info(f"Root mean square deviation (RMSD): {rmsd:.3}")
        st.info(f"Mean absolute error (MAE): {mae:.3f}")

    except Exception:
        st.error("**Error**: Enter correct path of file (.csv)!")
        sys.exit("csv file is not provided.")

    st.subheader("Shapiro-Wilk (W) Test")

    with st.expander("Theoretical Information"):

        st.write(
            """For a given sample size, the Shapiro-Wilk *W* statistics has been shown to 
have good properties. The values of *W* range between 0 and 1, and values of *W* ≃ 1 are in 
favor of the normality of the sample. If *W* lies below a critical value $W_{c}$
depending on the sample size and the chosen level of type I errors *α* (typically
0.05), the normality hypothesis cannot be assumed to hold."""
        )
    try:
        shapiro_wilk_w, p_value = shapiro(residual_error)

        col1, col2 = st.columns(2)
        col1.metric("W", np.round(shapiro_wilk_w, 4))
        col2.metric("Pr", np.round(p_value, 3))

        if 0.05 < p_value:
            st.info("So, error distribution is normal (*Pr* > 0.05).")

    except Exception:
        st.error("**Error**: Enter correct path of file (.csv)!")

    st.subheader("Skewness and Kurtosis Tests")

    with st.expander("Theoretical Information"):

        st.write(
            """Two other statistics are helpful in characterizing the departure from normality. The skewness (*Skew*), or
third standardized moment of the distribution, quantifies its asymmetry (*Skew* = 0 for a symmetric
distribution). The kurtosis (*Kurt*), or fourth standardized moment, quantifies the concentration of data in
the tails of the distribution. Kurtosis of a normal distribution is equal to 3; distributions with excess kurtosis
(*Kurt > 3*) are called *leptokurtic*; those with *Kurt < 3* are named *platykurtic*."""
        )
    try:
        skewness_value = skew(residual_error)
        kurtosis_value = kurtosis(residual_error, fisher=False)

        col1, col2 = st.columns(2)
        col1.metric("Skew", np.round(skewness_value, 4))
        col2.metric("Kurtosis", np.round(kurtosis_value, 4))

        if skewness_value != 0 and kurtosis_value > 3:
            st.info("Error distribution is asymmetric and leptokurtic.")

        if skewness_value != 0 and kurtosis_value < 3:
            st.info("Error distribution is asymmetric and platykurtic.")

    except Exception:
        st.error("**Error**: Enter correct path of file (.csv)!")

    st.subheader("$95^{th}$ quantile of the absolute errors distribution")

    with st.expander("Theoretical Information"):

        st.write(
            """For error distributions which are non symmetric (*Skew* not 0), quantifying the accuracy by a single
dispersion-related statistic is not reliable, and one should provide probability intervals or accept to lose
information on the sign and use a statistic based on absolute errors, such as $Q_{95}$
(the 95th percentile of the absolute error distribution, gives the amplitude of errors that there is a 5%
probability to exceed)."""
        )
    try:

        q95_value = np.percentile(np.abs(residual_error), 95)

        st.metric("Q95", np.round(q95_value, 4))

    except Exception:
        st.error("**Error**: Enter correct path of file (.csv)!")

    st.subheader("Normal Quantile-Quantile Plot")

    with st.expander("Theoretical Information"):
        st.write(
            """It might also be useful to assess normality by visual tools: normal quantile-quantile plots (*QQ-plots*),
where the quantiles of the scaled and centered errors sample is plotted against the theoretical quantiles of a
standard normal distribution (in the normal case, all points should lie over the unit line);"""
        )

    def plot_qqplot(residual, color="g", save=False):

        fig = plt.figure(figsize=(10, 8))
        x, y = range(-4, 5), range(-4, 5)
        quantile = probplot(residual, plot=None, fit=False)
        plt.plot(x, y, "--", label="Standard Normal Distribution")
        plt.plot(quantile[0], quantile[1], color, label="Sample")
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.xlabel("Theoretical Quantile", fontsize=14)
        plt.ylabel("Sample Quantile", fontsize=14)
        plt.title("Normal Quantile-Quantile Plot", fontsize=16)
        if save:
            plt.savefig("qqplot.png", dpi=300)
        plt.legend()
        plt.show()

        return fig

    save = st.checkbox("Save QQ-plot in .png")

    color = st.color_picker("Choose color for QQ-plot:", value="#66CDAA")

    fig = plot_qqplot(residual_error, color, save)

    st.pyplot(fig)

    st.warning(
        "**Caution**: To download plot, first check *Save QQ-plot in .png* then the download button appears."
    )

    if save:

        with open("qqplot.png", "rb") as file:

            st.download_button(
                "Download outliers plot", data=file, file_name="qqplot.png"
            )

    st.subheader("Histogram of Error Distribution")

    with st.expander("Theoretical Information"):
        st.write(
            """comparison of the histogram of errors with a gaussian curve having the same mean, 
estimated by the mean signed error(*MSE*), and same standard deviation, estimated by the *RMSD*."""
        )

    def histogram_and_normal_plot(
        mse, rmsd, residual, color1="darkcyan", color2="orangered", save=False
    ):

        mu, sigma = mse, rmsd
        fig = plt.figure(figsize=(10, 8))
        _, bins, _ = plt.hist(residual, bins=20, density=True, color=color1)
        plt.plot(
            bins,
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-((bins - mu) ** 2) / (2 * sigma ** 2)),
            linewidth=2,
            color=color2,
        )
        plt.xlabel("Residual Error", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.title("Histogram of Residual Errors", fontsize=16)
        if save:
            plt.savefig("residual_histogram.png", dpi=300)
        plt.show()

        return fig

    save = st.checkbox("Save histogram plot in .png.")

    color1 = st.color_picker("Choose color for histogram plot:", value="#5F9EA0")
    color2 = st.color_picker("Choose color for guassian plot:", value="#DC143C")

    fig_hist = histogram_and_normal_plot(
        mse, rmsd, residual_error, color1, color2, save
    )

    st.pyplot(fig_hist)

    st.warning(
        "**Caution**: To download plot, first check *Save histogram plot in .png* then the download button appears."
    )

    if save:

        with open("residual_histogram.png", "rb") as file:

            st.download_button(
                "Download outliers plot", data=file, file_name="residual_histogram.png"
            )

    st.subheader("Outliers Plot")

    with st.expander("Theoretical Information"):
        st.write(
            """There is no unique method to identify outliers for a non-normal distribution. One might, for instance, use
visual tools, such as *QQ-plots*, or automatic selection tools, such as selecting points for which the
absolute error is larger than the 95th percentile ($Q_{95}$), or another percentile corresponding to a predefined
error threshold."""
        )

    def outliers_plot(
        nonoutliers_df,
        outliers_df,
        color1="deepskyblue",
        color2="firebrick",
        save=False,
    ):

        fig = plt.figure(figsize=(10, 8))
        plt.scatter(
            nonoutliers_df["Target"], nonoutliers_df["Residual"], s=50, c=color1
        )
        plt.scatter(
            outliers_df["Target"], outliers_df["Residual"], s=50, c=color2, marker="^"
        )
        plt.xlabel("Experimental", fontsize=14)
        plt.ylabel("Residual Error", fontsize=14)
        plt.title("Error Distribution", fontsize=16)
        if save:
            plt.savefig("error_distribution.png", dpi=300)
        plt.show()

        return fig

    percentile = st.slider("Choose a specific percentile", value=90, step=5)

    df["ABS"] = (df["Predicted"] - df["Target"]).abs()
    df["Residual"] = residual_error
    q_value = np.percentile(np.abs(residual_error), int(percentile))
    outliers_df = df[df["ABS"] > q_value]
    nonoutliers_df = df.drop(outliers_df.index, axis=0)

    save = st.checkbox("Save outliers plot in .png")

    color1 = st.color_picker("Choose color for non-outliers:", value="#00CDCD")
    color2 = st.color_picker("Choose color for outliers:", value="#FF4040")

    fig_outliers = outliers_plot(nonoutliers_df, outliers_df, color1, color2, save)

    st.pyplot(fig_outliers)

    st.warning(
        "**Caution**: To download plot, first check *save outliers plot in .png* then the download button appears."
    )

    if save:

        with open("error_distribution.png", "rb") as file:

            st.download_button(
                "Download outliers plot", data=file, file_name="error_distribution.png"
            )

    st.write("""All theoretical information in this section is extracted from [3].""")

if add_selectbox == "3-Prediction":

    st.subheader("Binding Affinity Prediction")
    st.write("Binding affinity of complexes are predicted using a ML-Score.")
    st.warning(
        "**Caution 1**: All complexes have to turn to numerical representation using **Feature Generation** mode."
    )

    st.warning(
        """**Caution 2**: If you want to use GB-Score as a predictor, ligand and protein structures should be 
        hydrogenated. Use **Add Hydrogen** mode for this. Then, features should be generated using **n=2**, 
        **$d_{cutoff}$=12 A**. Also, **columns_pdbbind_2019.txt**, **mean_pdbbind_2019.csv**, and
        **std_pdbbind_2019.csv** files should be used for **Features name**, **Mean of features**, and 
        **STD of features** respectively."""
    )
    
    gpu = st.checkbox(
        "Use GPU accelerator during prediction.",
        value=False,
        help="XGBT uses GPU to accelerate predicting procedure.",
    )

    path_ml_score = st.text_input(
        "A ML saved model",
        value="Example/model/gb_score_cpu.json",
        help="A ML saved model based on XGBoost in .joblib. e.g /Example/model/gb_score_cpu.json",
    )

    assert os.path.exists(path_ml_score), "Error: File doesn't exist."
    assert (
        os.path.splitext(path_ml_score)[1][1:].strip().lower() == "json"
    ), "File extension should be json."

    path_columns = st.text_input(
        "Features name",
        value="Example/files/columns_pdbbind_2019.txt",
        help="""A file (.txt) contains name of all not discarded columns during
        preprocessing. Generated during **Model Training** as *columns.txt*. e.g /Example/files/columns_pdbbind_2019.txt""",
    )

    assert os.path.exists(path_columns), "Error: File doesn't exist."
    assert (
        os.path.splitext(path_columns)[1][1:].strip().lower() == "txt"
    ), "File extension should be text."

    path_mean = st.text_input(
        "Mean of features",
        value="Example/files/mean_pdbbind_2019.csv",
        help="""A csv file contain all mean for features. Generated during **Model Training** as *mean.csv*. e.g /Example/files/mean_pdbbind_2019.csv""",
    )

    assert os.path.exists(path_mean), "Error: File doesn't exist."
    assert (
        os.path.splitext(path_mean)[1][1:].strip().lower() == "csv"
    ), "File extension should be csv."

    path_std = st.text_input(
        "STD of features",
        value="Example/files/std_pdbbind_2019.csv",
        help="A csv file contain all std for features. Generated during **Model Training** as *std.csv*. e.g /Example/files/std_pdbbind_2019.csv",
    )

    assert os.path.exists(path_std), "Error: File doesn't exist."
    assert (
        os.path.splitext(path_std)[1][1:].strip().lower() == "csv"
    ), "File extension should be csv."

    path_x_test = st.text_input(
        "Generated features of complex",
        value="Example/files/x_test_set.csv",
        help="A csv file contains test set features. e.g /Example/files/x_test_set.csv",
    )

    assert os.path.exists(path_x_test), "Error: File doesn't exist."
    assert (
        os.path.splitext(path_x_test)[1][1:].strip().lower() == "csv"
    ), "File extension should be csv."

    y_pred_filename = st.text_input(
        "Prediction filename",
        value="Example/files/pred_test_set.csv",
        help="Filename for saving prediction with .csv extension. e.g /Example/files/pred_test_set.csv",
    )

    assert (
        os.path.splitext(y_pred_filename)[1][1:].strip().lower() == "csv"
    ), "File extension should be csv."

    path_y_test = st.text_input(
        "Experimental binding affinity value of test set.",
        value=None,
        help="Label of the test set in .csv.",
    )

    execute = st.button(
        "Start predicting operation", help="Press the button to start prediction.",
    )

    if execute:

        start = time.time()

        with st.spinner("In progress..."):

            y_df = predict(
                path_ml_score,
                path_columns,
                path_mean,
                path_std,
                path_x_test,
                y_pred_filename,
                path_y_test,
                gpu,
            )

        end = time.time()

        seconds = end - start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        st.success(f"Procedure is completed at: {h:.2f} h, {m:.2f} min, {s:.2f} sec!")

        st.info("Results:")

        st.dataframe(y_df, width=400, height=600)

if add_selectbox == "5-Add Hydrogen":

    def add_hydrogen(path_structures):

        path = Path(path_structures)

        for item in list(path.iterdir()):

            ligand = next(item.glob("*.mol2")).as_posix()
            protein_pdb = next(item.glob("*.pdb")).as_posix()
            protein_pqr = protein_pdb.replace(".pdb", ".pqr")
            protein_log = protein_pdb.replace(".pdb", ".log")

            subprocess.run(
                f"obabel -imol2 {ligand} -omol2 -O {ligand} -p 7.4", shell=True
            )
            subprocess.run(
                f"pdb2pqr30 --ff=AMBER  --with-ph=7.4 {protein_pdb} {protein_pqr}",
                shell=True,
            )
            #--chain ' '
            subprocess.run(
                f"obabel -ipqr {protein_pqr} -opdb -O {protein_pdb}", shell=True
            )

            if os.path.isfile(protein_pqr):
                os.remove(protein_pqr)

            if os.path.isfile(protein_log):
                os.remove(protein_log)

    st.write(
        """Add hydrogens to ligand and protein at pH=7.4 using 
[PDB2PQR](https://www.cgl.ucsf.edu/chimera/docs/ContributedSoftware/apbs/pdb2pqr.html) and 
[Openbabel](http://openbabel.org/wiki/Main_Page)."""
    )

    with st.expander("Caution about directory"):

        st.warning(
            "**Caution**: Directory of structures should be like the following picture:"
        )

        image2 = Image.open("tree.png")

        st.image(image2)

    directory = st.text_input(
        "Enter directory of your complex structures:",
        value="Example/structures",
        help="Indicating path of complex structures. e.g Example/structures",
    )

    assert os.path.isdir(directory), "Enter valid directory."

    path = Path(directory)

    for folder in path.iterdir():

        if not all(
            file.endswith(".mol2") or file.endswith(".pdb")
            for file in os.listdir(folder)
        ):

            st.error(
                """**Error**: Please correct your provided dicrectory. It contains files which their extension is 
                not.mol2 or .pdb."""
            )

    execute = st.button(
        "Start operation", help="Press the button to start prediction.",
    )

    if execute:

        start = time.time()

        with st.spinner("In progress..."):

            add_hydrogen(directory)

        end = time.time()

        seconds = end - start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        st.success(f"Procedure is completed at: {h:.2f} h, {m:.2f} min, {s:.2f} sec!")


st.sidebar.write(
    "Designed by [*Milad Rayka*](https://github.com/miladrayka), MIT License "
)
