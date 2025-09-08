from setuptools import setup, find_packages

setup(
    name="dice_art",
    version="0.1.0",
    description="A Python project for generating dice art.",
    author="Jason De Melo",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'Pillow',
        'opencv-python',
        'click',
        'pytest'
    ],
)