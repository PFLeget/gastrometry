#!/usr/bin/env python

"""Setup script."""

import glob
from setuptools import setup, find_packages

# Package name
name = 'gastrometry'

# Packages (subdirectories in clusters/)
packages = find_packages()

# Scripts (in scripts/)
scripts = glob.glob("scripts/*.py")

package_data = {}

setup(name=name,
      description=("gastrometry"),
      classifiers=["Topic :: Scientific :: Astronomy",
                   "Intended Audience :: Science/Research"],
      author="PFLeget",
      packages=packages,
      scripts=scripts)
