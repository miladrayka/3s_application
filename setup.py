
from setuptools import find_packages, setup


setup(
    name="threesss",
    author="Milad Rayka",
    author_email="milad.rayka@yahoo.com",
    description="Straight forwarding making ML-based scoring function.",
    version="1.0",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    url="www.github.com/miladryaka",
    install_requires=["numpy==1.24.0rc2", 
                      "scipy==1.10.0rc1", 
                      "pandas==1.5.2", 
                      "biopandas==0.4.1", 
                      "streamlit==1.16.0", 
                      "jupyterlab==4.0.0a31",
                      "matplotlib==3.6.2",
		              "seaborn==0.12.1",
                      "scikit-learn==1.2.0",
                      "xgboost==1.7.2",
                      "progressbar2==4.3b0",
                      "pdb2pqr==3.5.2"],
    platforms=["Windows"],
    python_requires="==3.9",
)
