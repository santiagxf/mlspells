import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlspells",
    version="0.0.1",
    author="Facundo Santiago",
    description="An army of ML tools to improve productivy",
    url = 'https://github.com/santiagxf/mlspells',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      
    python_requires='>=3.8',                
    packages=setuptools.find_packages(where='src', exclude=("tests",)),   
    package_dir={'':'src'},  
    include_package_data=True,   
    install_requires=[
        'pandas',
        'numpy',
    ]
)