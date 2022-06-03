from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

VERSION = '0.0.1'
DESCRIPTION = 'Dexa'

setup(
    name="dexa",
    version=VERSION,
    author="Hugo Cruz",
    author_email="<huggcruzz@gmail.com>",
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/hugocruzz/Dexa',
    install_requires=[
        'pandas==1.0.1'
        'numpy==1.18.1'
        'netCDF4==1.5.3'
        'PyYAML==6.0'
        'matplotlib==3.1.3'
        'seaborn==0.11.1'
        'scipy==1.5.4'
        'dask==2022.2.0'
        'xarray==0.19.0'
        'seawater==3.3.4'
    ],
    license="MIT",
    keywords=['python', 'dexa', 'Lake analyzer', 'environmental data', 'Physical properties'],
)