#!/bin/bash

pip uninstall -y numpy
rm -rf "<site-packages>/numpy"        # remove whatever survived the uninstall
rm -rf "<site-packages>"/numpy-*.dist-info
pip install "numpy==1.26.4"