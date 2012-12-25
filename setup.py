from setuptools import setup, find_packages
import os

setup(
    name='flipper',
    version='1.0',
    author='Nick Hand',
    author_email='nicholas.adam.hand@gmail.com',
    packages=find_packages(),
    scripts=['bin/' + script for script in os.listdir('bin')],
    description='flipper module'
)