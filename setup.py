from setuptools import setup, find_packages

setup(
    name="glycobertbart",
    version="0.1.0",
    author="Ejas",
    author_email="ejasalth@buffalo.edu",
    description="BERT and BART models for glycan structure prediction from LC-MSMS data",
    url="https://github.com/ejasalth/glycobertbart",
    packages=find_packages(),

install_requires=[
        "CandyCrunch[draw]==0.6.0,
        "glycowork~=1.6.1",
        "torch>2.1,",
        "transformers>=4.50.0,<5.0.0",
        "huggingface-hub>0.3,<0.4",
        "pandas",
        "numpy",
        "pyopenms~=3.4",
    ],
    
    include_package_data=True,
    package_data={'glycobertbart': ['*.pkl', '*.json', '*pt']}, 
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    long_description=open("README.md", encoding="utf-8").read(),                      
    long_description_content_type="text/markdown"
  
)
