#!/bin/bash

#This code is used to build and publish PySummarization package
#pip install --user --upgrade setuptools wheel twine gitpython

dirty_dirs=("build" "dist" "*.egg-info")

for x in ${dirty_dirs[@]}; do
	[ -e $x ] && echo $x
done
for x in ${dirty_dirs[@]}; do
	[ -e $x ] && rm -rf $x
done

#generating distribution archives
#python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel  # this generating a distribution file under dist/

#uploading the distribution archives
# find the index you want to upload your package to, called testpypi
#python3 -m pip install --user --upgrade twine
#python3 -m twine upload --repository testpypi dist/*

#testpypi: https://test.pypi.org/

#real production: https://pypi.org


#How to use the package
#pip install -i https://test.pypi.org/simple/ example-pkg-hw20201218