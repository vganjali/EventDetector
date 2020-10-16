from io import open
from setuptools import setup

setup(
    name='CWT Event Detector',
    version='0.1.1',
    url='https://github.com/vganjali/EventDetector',
    license='MIT',
    author='Vahid Ganjalizadeh',
    author_email='vganjali@ucsc.edu',
    description='Analysis tool for fluorescent cytometry and single-molecule detection using CWT',
    long_description=''.join(open('README.md', encoding='utf-8').readlines()),
    long_description_content_type='text/markdown',
    keywords=['cwt','SMD','gui'],
    packages=['cwted'],
    include_package_data=True,
    install_requires=['requests>=2.22.0','h5py>=2.10.0','matplotlib>=3.2.2','pyside2>=5.13.2','pyqtgraph>=0.11.0','numpy>=1.18.5','pandas>=1.0.5','scipy>=1.5.0'],
    python_requires='>=3.7',
    classifiers=[
        'License :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
    ],
)
