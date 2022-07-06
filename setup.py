from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

VERSION = '0.0.13'
DESCRIPTION = 'pylake'

setup(
    name="pylake",
    version=VERSION,
    author="Hugo Cruz",
    author_email="<huggcruzz@gmail.com>",
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/hugocruzz/pylake',
    install_requires=[
        'numpy>=1.19.5',
        'pandas>=1.1.5',
        'PyYAML>=6.0',
        'scipy>=1.5.4',
        'dask>=2022.02.1',
        'xarray>=0.19.0',   
        'seawater>=3.3.4'
    ],
    license="MIT",
    keywords=['python', 'pylake', 'Lake analyzer', 'environmental data', 'Physical properties'],
)