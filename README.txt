distcan
=======

Probability **dist**\ ributions for python in their **can**\ onical
form. Documentation (TODO: link)

``scipy.stats`` is the go to library for working with probability
distributions in python. It is an impressive package that exposes an
*internally consistent* API for working with almost 100 distributions.
But, there are some shortcomings...

-  Instead of using the common names for the parameters of distributions
   (e.g. Normal distribution mean and standard deviation being named mu
   and sigma), ``scipy.stats`` has keyword arguments (or combinations of
   them) ``loc``, ``scale``, and ``shape`` *assume* the roles of
   canonical parameters
-  Related to the non-conventional parameter naming, the documentation
   displays expressions for the pdf that often doesn't match the
   canonical form of the pdf easily found online or in standard
   references. This makes it difficult to tell exactly what distribution
   you are working with
-  Some distributions are included in ``scipy.stats``, but under a
   different name a different documented form for the pdf. For example,
   to create an InverseGamma(5, 6) distribution, you would call
   ``scipy.stats.invgamma(5, scale=6)``

Enter ``distcan``
-----------------

The ``distcan`` library aims to address these problems in an easily
extensible way. Some goals of this project are

-  Represent probability distributions in their canonical form, with
   parameters given their standard names
-  Expose an API that is encompasses functionality in ``scipy.stats``
   and
   ```Distributions.jl`` <https://github.com/JuliaStats/Distributions.jl>`__
   (a Julia package that motivated the creation of ``distcan``), with
   naming conventions that are consistent for users of both packages
-  Have documentation that accurately describes the distribution being
   used

By leveraging the great code in ``scipy.stats``, we are well on our way
to completing these goals.

Functionality
~~~~~~~~~~~~~

All the functionality of ``scipy.stats``, plus a few other convenience
methods, is exposed by each distribution. This includes the following
methods:

-  ``pdf``: evaluate the probability density function
-  ``logpdf``: evaluate the log of the pdf
-  ``cdf``: evaluate the cumulative density function
-  ``logcdf``: evaluate the log of the cdf
-  ``rvs``: draw random samples from the distribution
-  ``moment``: evaluate nth non-central moment
-  ``stats``: some statistics of the RV (such as mean, variance,
   skewness, kurtosis)
-  ``fit`` (when available in scipy.stats): return the maximum
   likelihood estimators of the distribution, given data
-  ``sf`` (also given name ccdf): compute the survival function (or
   complementary cumulative density function)
-  ``logsf`` (also given name logccdf): compute the log of the survival
   function (or complementary cumulative density function)
-  ``isf``: compute the inverse of the survival function (or
   complementary cumulative density function)
-  ``ppf`` (also give name quantile): compute the percent point function
   (or quantile), which is the inverse of the cdf. This is commonly used
   to compute critical values.
-  ``loglikelihood`` (not in scipy): the loglikelihood of the
   distribution with respect to all the samples in x
-  ``invlogcdf`` (not in scipy): evaluate inverse function of the logcdf
-  ``cquantile`` (not in scipy): evaluate the complementary quantile
   function. Equal to ``d.ppf(1-x)`` for x in (0, 1). Could be used to
   compute the lower critical values of a distribution
-  ``invlogccdf`` (not in scipy): evaluate inverse function of the
   logccdf

Additionally, each distribution has the following properties (accessed
as ``dist_object.property_name`` -- i.e. without parenthesis):

-  ``mean``: the mean of the distribution
-  ``var``: the var of the distribution
-  ``std``: the std of the distribution
-  ``skewness``: the skewness of the distribution
-  ``kurtosis``: the kurtosis of the distribution
-  ``median``: the median of the distribution
-  ``mode``: the mode of the distribution
-  ``isplaykurtic``: boolean indicating if kurtosis is greater than zero
-  ``isleptokurtic``: boolean indicating if kurtosis is less than zero
-  ``ismesokurtic``: boolean indicating if kurtosis is equal to zero
-  ``entropy``: the entropy of the distribution
-  ``params`` (not in scipy): return a tuple of the distributions
   parameters

Contributors
------------

-  Spencer Lyon (spencer.lyon@stern.nyu.edu)

