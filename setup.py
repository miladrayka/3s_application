
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
    install_requires=["numpy", 
                      "scipy", 
                      "pandas", 
                      "biopandas", 
                      "streamlit", 
                      "notebook",
                      "matplotlib",
		       "seaborn",
                      "scikit-learn",
                      "xgboost",
                      "progressbar2",
                      "pdb2pqr"],
    platforms=["Linux", "Windows"],
    python_requires=">=3.7",
)
