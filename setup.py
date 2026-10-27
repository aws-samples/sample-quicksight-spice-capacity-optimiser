# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from setuptools import setup, find_packages

setup(
    name="quicksight-spice-dataset-optimiser",
    version="1.0.0",
    description="Automated tool to identify and remove unused columns from Amazon QuickSight SPICE datasets",
    author="Amazon Web Services",
    license="MIT-0",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        # Must match requirements.txt. Below 1.43.64 the SDK does not recognise the
        # FileSource PhysicalTableMap variant, and it fails silently: DescribeDataSet
        # returns a hollow but well-formed response for file-upload and new-experience
        # datasets, so a healthy dataset reads as empty. This floor was 1.28.0, which
        # meant `pip install -e .` produced an environment that mis-reads real datasets.
        "boto3>=1.43.64",
    ],
    extras_require={
        # Pinned, and kept identical to requirements.txt, so a fresh install cannot pull
        # an untested release. pyyaml is used by the CloudFormation template tests.
        "dev": [
            "hypothesis==6.170.11",
            "pytest==9.1.1",
            "pytest-cov==7.1.0",
            "pyyaml==6.0.2",
        ]
    },
)
