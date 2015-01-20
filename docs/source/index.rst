.. distcan documentation master file, created by
   sphinx-quickstart on Tue Jan 20 10:13:30 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

distcan: probability distributions in canonical form
====================================================

distcan is a Python library that utilizes the mature code from ``scipy.stats`` to provide Python users with probability **dist** ributions in their **can** onical form. Some goals of this project are:

* Represent probability distributions in their canonical form, with parameters given their standard names
* Expose an API that is encompasses functionality in ``scipy.stats`` and `Distributions.jl <https://github.com/JuliaStats/Distributions.jl>`_ (a Julia package that motivated the creation of ``distcan``), with naming conventions that are consistent for users of both packages
* Have documentation that accurately describes the distribution being used

This project is still in the early stages, so check back often for changes.

.. toctree::
   :maxdepth: 2

   whatsnew
   intro
   univariate
   multivariate
   matrix



Indices and tables
==================

* :ref:``genindex``
* :ref:``modindex``
* :ref:``search``

