from setuptools import setup, find_packages

setup(
    name='chaos_evaluator',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
    ],
    author='Eason Suen',
    author_email='easonssuen@gmail.com',
    description='A package to evaluate chaos in time series data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/easonanalytica/chaos_evaluator',
)
