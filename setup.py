# Copyright (C) 2023 Wenbo Guo, G-Lab, Tsinghua University

from setuptools import setup, find_packages

with open("README.md", "r") as fh: 
	description = fh.read() 


setup( 
	name="scStateDynamics", 
	version="0.0.18", 
	author="Wenbo Guo", 
	author_email="gwb17@tsinghua.org.cn", 
    packages=find_packages(),
	description="A package to decipher the drug-responsive tumor cell state dynamics by modeling single-cell level expression changes", 
	long_description=description, 
	long_description_content_type="text/markdown", 
	url="https://github.com/wguo-research/scStateDynamics", 
	license='MIT', 
	python_requires='>=3.7', 
	install_requires=[
		"numpy",
    	"pandas",
    	"scipy",
    	"scikit-learn",
    	"torch",
    	"seaborn",
    	"matplotlib",
    	"scanpy",
    	"leidenalg",
    	"pyro-ppl",
    	"POT"] 
) 
