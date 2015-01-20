"""
Univariate distributions

@author : Spencer Lyon <spencer.lyon@stern.nyu.edu>
@date : 2015-01-07

"""
from math import sqrt, log, pi
import numpy as np
import scipy.stats as st
from scipy.special import gamma, gammaln
from .scipy_wrap import CanDistFromScipy

__all__ = ["InverseGamma", "Normal", "Gamma", "NormalInverseGamma"]

univariate_class_docstr = r"""
Construct a distribution representing {name} random variables. The pdf
of the distribution is given by

.. math::

    {pdf_tex}

Parameters
----------
{param_list}

Attributes
----------
{param_attributes}
mean :  scalar(float)
    mean of the distribution
std :  scalar(float)
    std of the distribution
var :  scalar(float)
    var of the distribution
skewness :  scalar(float)
    skewness of the distribution
kurtosis :  scalar(float)
    kurtosis of the distribution
median :  scalar(float)
    median of the distribution
mode :  scalar(float)
    mode of the distribution
isplatykurtic :  Boolean
    boolean indicating if d.kurtosis > 0
isleptokurtic :  bool
    boolean indicating if d.kurtosis < 0
ismesokurtic :  bool
    boolean indicating if d.kurtosis == 0
entropy :  scalar(float)
    entropy value of the distribution

"""

param_str = "{name} : {kind}\n    {descr}"


def _create_param_list_str(names, descrs, kinds="scalar(float)"):

    names = (names, ) if isinstance(names, str) else names
    names = (names, ) if isinstance(names, str) else names

    if isinstance(kinds, (list, tuple)):
        if len(names) != len(kinds):
            raise ValueError("Must have same number of names and kinds")

    if isinstance(kinds, str):
        kinds = [kinds for i in range(len(names))]

    if len(descrs) != len(names):
        raise ValueError("Must have same number of names and descrs")

    params = []
    for i in range(len(names)):
        n, k, d = names[i], kinds[i], descrs[i]
        params.append(param_str.format(name=n, kind=k, descr=d))

    return str.join("\n", params)


def _create_class_docstr(name, param_names, param_descrs,
                         param_kinds="scalar(float)",
                         pdf_tex=r"\text{not given}", **kwargs):
    param_list = _create_param_list_str(param_names, param_descrs,
                                        param_kinds)

    param_attributes = str.join(", ", param_names) + " : See Parameters"

    return univariate_class_docstr.format(**locals())

#  ------------  #
#  InverseGamma  #
#  ------------  #


class InverseGamma(CanDistFromScipy):

    _metadata = {
        "name": "InverseGamma",
        "pdf_tex": (r"p(x;\alpha,\beta)=\frac{\beta^{\alpha}}{\Gamma(\alpha)}"
                    + r"x^{-\alpha-1}\exp\left(-\frac{\beta}{x}\right)"),

        "cdf_tex": r"\frac{\Gamma(\alpha, \beta / x)}{\Gamma(\alpha)}",

        "param_names": ["alpha", "beta"],

        "param_descrs": ["Shape parameter (must be >0)",
                         "Scale Parameter (must be >0)"],

        "_str": "InverseGamma(alpha=%.5f, beta=%.5f)"}

    # set docstring
    __doc__ = _create_class_docstr(**_metadata)

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

        # set dist before calling super's __init__
        self.dist = st.invgamma(alpha, scale=beta)
        super(InverseGamma, self).__init__()

    @property
    def params(self):
        return (self.alpha, self.beta)


#  ------ #
#  Normal #
#  ------ #


class Normal(CanDistFromScipy):

    _metadata = {
        "name": "Normal",
        "pdf_tex": (r"p(x;\mu,\sigma)=\frac{1}{\sigma \sqrt{2\pi}}" +
                    r"e^{-\frac{(x-\mu)^2}{2\sigma^2}}"),

        "cdf_tex": (r"\frac{1}{2} \left[ 1 + \text{erf} " +
                    r"\left( \frac{x-\mu}{\sigma \sqrt{2}}\right)\right]"),

        "param_names": ["mu", "sigma"],

        "param_descrs": ["mean of the distribution",
                         "Standard deviation of the distribution"],

        "_str": "Normal(mu=%.5f, sigma=%.5f)"}

    # set docstring
    __doc__ = _create_class_docstr(**_metadata)

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        # set dist before calling super's __init__
        self.dist = st.norm(mu, scale=sigma)
        super(Normal, self).__init__()

    @property
    def params(self):
        return (self.mu, self.sigma)


#  ----- #
#  Gamma #
#  ----- #

class Gamma(CanDistFromScipy):

    _metadata = {
        "name": "Gamma",
        "pdf_tex": (r"p(x;\alpha,\beta)=\frac{x^{\alpha-1}e^{-x/\beta}}" +
                    r"{\Gamma(\alpha)\beta^{\alpha}}"),

        "cdf_tex": (r"\frac{\gamma(\alpha, \beta x)}{\Gamma(\alpha)}" + "\n\n"
                    + r"where :math:`\gamma(\cdot)` is the incomplete"
                    + " gamma function"),

        "param_names": ["alpha", "beta"],

        "param_descrs": ["Shape parameter", "Scale Parameter"],

        "_str": "Gamma(alpha=%.5f, beta=%.5f)"}

    # set docstring
    __doc__ = _create_class_docstr(**_metadata)

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

        # set dist before calling super's __init__
        self.dist = st.gamma(alpha, scale=beta)
        super(Gamma, self).__init__()

    @property
    def params(self):
        return (self.alpha, self.beta)


# ########################################################################## #
# Below we have other distributions that are not a part of scipy.stats       #
# ########################################################################## #


#  ------------------ #
#  NormalInverseGamma #
#  ------------------ #

# TODO Clean up NIG docstrings

_nig_pdf_doc = r"""
Evaluate the probability density function, which is defined as

.. math::

    {pdf_tex}

Parameters
----------
mu : array_like or scalar(float)
    The mu point(s) (N part of NIG) at which to evaluate the pdf
sig2 : array_like or scalar(float)
    The sigma point(s) (IG part of NIG) at which to evaluate the pdf

Returns
-------
out : {ret1_type}
    The pdf of the distribution evaluated at (mu, sig2) pairs
"""


class NormalInverseGamma(object):

    def __init__(self, mu, v0, shape, scale):
        self.mu = mu
        self.v0 = v0
        self.shape = shape
        self.scale = scale

        # define docstring arguments
        pdf_tex = r"p(x;\alpha,\beta)=\frac{x^{\alpha-1}e^{-x/\beta}}"
        pdf_tex += r"{\Gamma(\alpha)\beta^{\alpha}}"
        cdf_tex = r"\frac{\gamma(\alpha, \beta x)}{\Gamma(\alpha)}" + "\n\n"
        cdf_tex += r"where :math:`\gamma(\cdot)` is the incomplete"
        cdf_tex += " gamma function"

        self._str = "Normal(mu=%.5f, sigma=%.5f, alpha=%.5f, beta=%.5f)"

    def pdf(self, x, sig2):
        m, v, sh, sc = self.mu, self.v0, self.shape, self.scale
        Zinv = sc**sh / gamma(sh) / sqrt(v * 2*pi)
        return (Zinv * 1./(np.sqrt(sig2) * sig2**(sh+1.)) *
                np.exp(-sc/sig2 - 0.5/(sig2*v)*(x-m)**2.0))

    def logpdf(self, x, sig2):
        m, v, sh, sc = self.mu, self.v0, self.shape, self.scale
        lZinv = sh*log(sc) - gammaln(sh) - 0.5*(log(v) + log(2*pi))
        return (lZinv - 0.5*np.log(sig2) - (sh+1.)*np.log(sig2) -
                sc/sig2 - 0.5/(sig2*v)*(x-m)**2)

    @property
    def mode(self):
        return self.mu, self.scale / (self.shape + 1.0)

    @property
    def mean(self):
        sig2 = self.scale / (self.shape - 1.0) if self.shape > 1.0 else np.inf
        return self.mu, sig2

    def _rand1(self):
        sig2 = InverseGamma(self.shape, self.scale).rand()

        if sig2 <= 0.0:
            sig2 = np.finfo(float).resolution  # equiv to julia eps()

        mu = Normal(self.mu, sqrt(sig2 * self.v0)).rand()
        return mu, sig2

    def rand(self, n=1):
        if n == 1:
            return self._rand1()
        else:
            out = np.empty((n, 2))
            for i in range(n):
                out[i] = self._rand1()

            return out


if __name__ == '__main__':
    nig = NormalInverseGamma(0.0, 1.0, 5.0, 6.0)
    print(nig.pdf(1.0, 3.0))
