from setuptools import find_packages, setup

setup(
    name="sgcd",
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'numpy',
        'mne',
        'matplotlib',
        'pandas',
        'scipy',
        'pingouin',
        'statsmodels',
        'scikit-learn',
        'autoreject',
        'opencv-python',  
        'scikit-image',  
        'jupyter',
    ],
    author="Aylin Kallmayer",
    description="Codebase for 'Object representations reflect hierarchical scene structure and depend on high-level visual, semantic, and action information'", 
    url="https://osf.io/fb3sj/"
)
