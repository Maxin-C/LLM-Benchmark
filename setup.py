#!/usr/bin/env python3
"""
EASE Benchmark Setup Script
专家锚定的自适应仿真评估框架
"""

from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='ease-benchmark',
    version='1.0.0',
    description='Expert-Anchored Adaptive Simulation Evaluation Framework',
    author='EASE Team',
    author_email='ease@example.com',
    url='https://github.com/ease-benchmark/ease',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'ease=main:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Healthcare',
    ],
    python_requires='>=3.9',
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.json'],
    },
)
