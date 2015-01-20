"""
Common distributions with standard parameterizations in Python

@author : Spencer Lyon <spencer.lyon@stern.nyu.edu>
@date : 2014-12-31 15:59:31

"""
from math import sqrt
import numpy as np

__all__ = ["CanDistFromScipy"]

pdf_docstr = r"""
Evaluate the probability density function, which is defined as

.. math::

    {pdf_tex}

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate the pdf

Returns
-------
out : {ret1_type}
    The pdf of the distribution evaluated at x

Notes
-----
For applicable distributions, equivalent to calling `d__dist_name(x,
*args, log=0)` from R

"""

logpdf_docstr = r"""
Evaluate the log of the pdf, where the pdf is defined as

.. math::

    {pdf_tex}

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate the log of the pdf

Returns
-------
out : {ret1_type}
    The log of pdf of the distribution evaluated at x

Notes
-----
For applicable distributions, equivalent to calling `d__dist_name(x,
*args, log=1)` from R

"""

cdf_docstr = r"""
Evaluate the cumulative density function

.. math::

    {cdf_tex}

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate the cdf

Returns
-------
out : {ret1_type}
    The cdf of the distribution evaluated at x

Notes
-----
For applicable distributions, equivalent to calling `p__dist_name(x,
*args, lower.tail=1, log.p=0)` from R

"""

logcdf_docstr = r"""
Evaluate the log of the cdf, where the cdf is defined as

.. math::

    {cdf_tex}

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate the log of the cdf

Returns
-------
out : {ret1_type}
    The log of cdf of the distribution evaluated at x

Notes
-----
For applicable distributions, equivalent to calling `p__dist_name(x,
*args, lower.tail=1, log.p=1)` from R

"""

rvs_docstr = r"""
Draw random samples from the distribution

Parameters
----------
size : tuple
    A tuple specifying the dimensions of an array to be filled with
    random samples

Returns
-------
out : {ret1_type}
    The random sample(s) requested

"""

sf_docstr = r"""
Compute the survival function (or complementary cumulative density
function) of the distribution at given points. This is defined as

.. math::

    sf(x) = ccdf(x) = 1 - cdf(x)

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate the sf (ccdf)

Returns
-------
out : {ret1_type}
    One minus the cdf of the distribution evaluated at x

Notes
-----
For applicable distributions, equivalent to calling `p__dist_name(x,
*args, lower.tail=0, log.p=0)` from R

"""

logsf_docstr = r"""
Compute the log of the survival function (or complementary cumulative
density function) of the distribution at given points. This is defined
as

.. math::

    \log(sf(x)) = \log(ccdf(x)) = \log(1 - cdf(x))

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate the log of the sf (ccdf)

Returns
-------
out : {ret1_type}
    Log of one minus the cdf of the distribution evaluated at x

Notes
-----
For applicable distributions, equivalent to calling `p__dist_name(x,
*args, lower.tail=1, log.p=1)` from R

"""

isf_docstr = r"""
Compute the inverse of the survival function (or complementary
cumulative density function) of the distribution at given points. This
is commonly used to find critical values of a distribution

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate the log of the sf (ccdf)

Returns
-------
out : {ret1_type}
    Log of one minus the cdf of the distribution evaluated at x

Examples
--------
>>> d.isf([0.1, 0.05, 0.01])  # upper tail critical values

"""

ppf_docstr = r"""
Compute the percent point function (or quantile), which is the inverse
of the cdf. This is commonly used to compute critical values.

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate the log of the sf (ccdf)

Returns
-------
out : {ret1_type}
    Log of one minus the cdf of the distribution evaluated at x

Examples
--------
>>> d.isf([0.1, 0.05, 0.01])  # upper tail critical values

Notes
-----
The ppf(x) = ccdf(1 - x), for x in (0, 1)

For applicable distributions, equivalent to calling `q__dist_name(x,
*args, lower.tail=1, log.p=0)` from R

"""

rand_docstr = r"""
Draw random samples from the distribution

Parameters
----------
*args : int
    Integer arguments are taken to be the dimensions of an array that
    should be filled with random samples

Returns
-------
out : {ret1_type}
    The random sample(s) requested

Examples
--------
>>> samples = d.rand(2, 2, 3); samples.shape  # 2, 3, 3 array of samples
(2, 3, 3)
>>> type(d.rand())
numpy.float64

"""

ll_docstr = r"""
The loglikelihood of the distribution with respect to all the samples
in x. Equivalent to sum(d.logpdf(x))

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate the log likelihood

Returns
-------
out : scalar
    The log-likelihood of the observations in x

"""

invlogcdf_docstr = r"""
Evaluate inverse function of the logcdf of the distribution at x

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate the inverse of the log of the cdf

Returns
-------
out : {ret1_type}
    The random variable(s) such that the log of the cdf is equal to x

Notes
-----
For applicable distributions, equivalent to calling `q__dist_name(x,
*args, lower.tail=1, log.p=1)` from R

"""

cquantile_docstr = r"""
Evaluate the complementary quantile function. Equal to `d.ppf(1-x)` for
x in (0, 1). Could be used to compute the lower critical values of a
distribution

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate 1 minus the quantile

Returns
-------
out : {ret1_type}
    The lower-tail critical values of the distribution

Notes
-----
For applicable distributions, equivalent to calling `q__dist_name(x,
*args, lower.tail=0, log.p=0)` from R

"""

invlccdf_docstr = r"""
Evaluate inverse function of the logccdf of the distribution at x

Parameters
----------
x : {arg1_type}
    The point(s) at which to evaluate the inverse of the log of the cdf

Returns
-------
out : {ret1_type}
    The random variable(s) such that the log of 1 minus the cdf is equal
    to x

Notes
-----
For applicable distributions, equivalent to calling `q__dist_name(x,
*args, lower.tail=0, log.p=1)` from R

"""

default_docstr_args = {"pdf_tex": r"\text{not given}",
                       "cdf_tex": r"\text{not given}",
                       "arg1_type": "array_like or scalar",
                       "ret1_type": "array_like or scalar"}


def _default_fit(self, x):
    msg = "If you would like to see this open an issue or submit a pull"
    msg += " request at https://github.com/spencerlyon2/distcan/issues"
    raise NotImplementedError(msg)


def _default_expect(self, x):
    msg = "If you would like to see this open an issue or submit a pull"
    msg += " request at https://github.com/spencerlyon2/distcan/issues"
    raise NotImplementedError(msg)


class CanDistFromScipy(object):

    def __init__(self):

        # assign scipy.stats.distributions.method_name to names I like
        # standard names
        self.pdf = self.dist.pdf
        self.logpdf = self.dist.logpdf
        self.cdf = self.dist.cdf
        self.logcdf = self.dist.logcdf
        self.rvs = self.dist.rvs
        self.moment = self.dist.moment
        self.stats = self.dist.stats

        # not all distributions have the following: fit, expect
        if hasattr(self.dist, "fit"):
            self.fit = self.dist.fit
        else:
            self.fit = _default_fit

        if hasattr(self.dist, "expect"):
            self.expect = self.dist.expect
        else:
            self.fit = _default_expect

        # survival function. Called the complementary cumulative
        # function (ccdf) in .jl
        self.sf = self.ccdf = self.dist.sf
        self.logsf = self.logccdf = self.dist.logsf
        self.isf = self.dist.isf

        # Distributions.jl calls scipy's ppf function quantile. I like that
        self.ppf = self.quantile = self.dist.ppf

        # set docstrings
        self._set_docstrings()
        self.__doc__ = "foobar"

    def _set_docstrings(self):
        fmt_args = default_docstr_args.copy()  # copy so ready for next use
        fmt_args.update(self._metadata)  # pull in data from subclass

        # define docstrings
        self.pdf.__func__.__doc__ = pdf_docstr.format(**fmt_args)
        self.logpdf.__func__.__doc__ = logpdf_docstr.format(**fmt_args)
        self.cdf.__func__.__doc__ = cdf_docstr.format(**fmt_args)
        self.logcdf.__func__.__doc__ = logcdf_docstr.format(**fmt_args)
        self.rvs.__func__.__doc__ = rvs_docstr.format(**fmt_args)

        # survival function stuff
        self.sf.__func__.__doc__ = sf_docstr.format(**fmt_args)
        self.ccdf.__func__.__doc__ = self.sf.__func__.__doc__
        self.logsf.__func__.__doc__ = logsf_docstr.format(**fmt_args)
        self.logccdf.__func__.__doc__ = self.logsf.__func__.__doc__
        self.isf.__func__.__doc__ = isf_docstr.format(**fmt_args)

        # ppf
        self.ppf.__func__.__doc__ = ppf_docstr.format(**fmt_args)
        self.quantile.__func__.__doc__ = self.ppf.__func__.__doc__

        # from distributions.jl
        self.rand.__func__.__doc__ = rand_docstr.format(**fmt_args)
        self.loglikelihood.__func__.__doc__ = ll_docstr.format(**fmt_args)
        self.invlogcdf.__func__.__doc__ = invlogcdf_docstr.format(**fmt_args)
        self.cquantile.__func__.__doc__ = cquantile_docstr.format(**fmt_args)
        self.invlogccdf.__func__.__doc__ = invlccdf_docstr.format(**fmt_args)

    def __str__(self):
        return self._metadata["_str"] % (self.params)

    def __repr__(self):
        return self.__str__()

    @property
    def mean(self):
        return self.dist.stats(moments="m")

    @property
    def var(self):
        return self.dist.stats(moments="v")

    @property
    def std(self):
        return sqrt(self.var)

    @property
    def skewness(self):
        return self.dist.stats(moments="s")

    @property
    def kurtosis(self):
        return self.dist.stats(moments="k")

    @property
    def median(self):
        return self.dist.median()

    @property
    def mode(self):
        return self.dist.ppf(0.5)

    @property
    def isplatykurtic(self):
        return self.kurtosis > 0

    @property
    def isleptokurtic(self):
        return self.kurtosis < 0

    @property
    def ismesokurtic(self):
        return self.kurtosis == 0.0

    @property
    def entropy(self):
        return float(self.dist.entropy())

    def rand(self, *args):
        return self.dist.rvs(size=args)

    def loglikelihood(self, x):
        return sum(self.logpdf(x))

    def invlogcdf(self, x):
        return self.quantile(np.exp(x))

    def cquantile(self, x):
        return self.quantile(1.0 - x)

    def invlogccdf(self, x):
        return self.quantile(-(np.exp(x) - 1.0))
