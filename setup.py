#!/usr/bin/python

##############################################

# Manoj M Wagle (USydney; MIT CSAIL; Broad Institute)

##############################################


from setuptools import setup, find_namespace_packages

setup(
    name="Phenoverse",
    version="0.1.1",
    packages=find_namespace_packages(),
    python_requires='>=3.12',
    install_requires=[
        'torch',
        'scanpy',
        'pandas',
        'numpy',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'phenoverse=phenoverse.Phenoverse:main',
        ],
    },
    include_package_data=True,
    description='Deep interpretable learning of sample representations for characterizing disease states in single-cell transcriptomics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KellisLab/Phenoverse',
    project_urls={
        'Documentation': 'https://kellislab.github.io/Phenoverse/',
    },
    author='Manoj M Wagle',
    author_email='manojmw@mit.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.12',
    ],
)
