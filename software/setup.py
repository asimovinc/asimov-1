from setuptools import find_packages
from distutils.core import setup

setup(
    name='asimov_rl',
    version='0.1.0',
    author='Asimov Inc.',
    license='BSD-3-Clause',
    packages=find_packages(),
    description='IsaacGym RL training and MuJoCo inference for Asimov v1 humanoid',
    install_requires=[
        # 'isaacgym',  # preview4 — install separately from NVIDIA
        'tensorboard',
        'numpy==1.23.5',
        'opencv-python',
        'mujoco==2.3.6',
        'mujoco-python-viewer',
        'matplotlib',
        'pygame',
    ],
)
