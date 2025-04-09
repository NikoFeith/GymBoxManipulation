from setuptools import setup, find_packages

setup(
    name='physics_block_rearrangement_env',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'pybullet',
        'numpy',
    ],
)