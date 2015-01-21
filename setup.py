from distutils.core import setup
import os

long_desc = 'Add a fallback short description here'
if os.path.exists('README.txt'):
    long_desc = open('README.txt').read()

# Write a versions.py file for class attribute
VERSION = "0.0.1"


def write_version_py(filename=None):
    doc = ("\"\"\"\n" +
           "This is a VERSION file and should NOT be manually altered"
           + "\n\"\"\"")
    doc += "\nversion = \"%s\"" % VERSION

    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), "distcan", "version.py")

    f = open(filename, "w")
    try:
        f.write(doc)
    finally:
        f.close()

write_version_py()

# Setup

setup(name="distcan",
      packages=["distcan"],
      version=VERSION,
      description="Probability distributions in their canonical form",
      author="Spencer Lyon",
      author_email="spencer.lyon@stern.nyu.edu",
      url="https://github.com/spencerlyon2/distcan",  # URL to the github repo
      keywords=["statistics", "distributions"],
      long_description=long_desc)
