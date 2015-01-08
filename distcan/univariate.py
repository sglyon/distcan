"""
Univariate distributions

@author : Spencer Lyon <spencer.lyon@stern.nyu.edu>
@date : 2015-01-07

"""
from math import sqrt, log, pi
import numpy as np
import scipy.stats as st
from scipy.special import gamma, gammaln
from scipy_wrap import CanDistFromScipy

__all__ = ["InverseGamma", "Normal", "Gamma", "NormalInverseGamma"]


#  ------------  #
#  InverseGamma  #
#  ------------  #

class InverseGamma(CanDistFromScipy):

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

        # set dist and _docstr_args attributes before calling super's __init__

        # define docstring arguments
        pdf_tex = r"p(x;\alpha,\beta)=\frac{\beta^{\alpha}}{\Gamma(\alpha)}"
        pdf_tex += r"x^{-\alpha-1}\exp\left(-\frac{\beta}{x}\right)"
        cdf_tex = r"\frac{\Gamma(\alpha, \beta / x)}{\Gamma(\alpha)}"

        self._docstr_args = {"pdf_tex": pdf_tex,
                             "cdf_tex": cdf_tex}

        self.dist = st.invgamma(alpha, scale=beta)
        super(InverseGamma, self).__init__()

        # set distribution name/params for __str__ and friends
        self._str = "InverseGamma(alpha=%.5f, beta=%.5f)"

    @property
    def params(self):
        return (self.alpha, self.beta)


#  ------ #
#  Normal #
#  ------ #

class Normal(CanDistFromScipy):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        # set dist and _docstr_args attributes before calling super's __init__
        # define docstring arguments
        pdf_tex = r"p(x;\mu,\sigma)=\frac{1}{\sigma \sqrt{2\pi}}"
        pdf_tex += r"e^{-\frac{(x-\mu)^2}{2\sigma^2}}"
        cdf_tex = r"\frac{1}{2} \left[ 1 + \text{erf} "
        cdf_tex += r"\left( \frac{x-\mu}{\sigma \sqrt{2}}\right)\right]"

        self._docstr_args = {"pdf_tex": pdf_tex,
                             "cdf_tex": cdf_tex}

        self.dist = st.norm(mu, scale=sigma)
        super(Normal, self).__init__()

        # set distribution name/params for __str__ and friends
        self._str = "Normal(mu=%.5f, sigma=%.5f)"

    @property
    def params(self):
        return (self.mu, self.sigma)


#  ----- #
#  Gamma #
#  ----- #

class Gamma(CanDistFromScipy):

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

        # set dist and _docstr_args attributes before calling super's __init__

        # define docstring arguments
        pdf_tex = r"p(x;\alpha,\beta)=\frac{x^{\alpha-1}e^{-x/\beta}}"
        pdf_tex += r"{\Gamma(\alpha)\beta^{\alpha}}"
        cdf_tex = r"\frac{\gamma(\alpha, \beta x)}{\Gamma(\alpha)}" + "\n\n"
        cdf_tex += r"where :math:`\gamma(\cdot)` is the incomplete"
        cdf_tex += " gamma function"

        self._docstr_args = {"pdf_tex": pdf_tex,
                             "cdf_tex": cdf_tex}

        self.dist = st.gamma(alpha, scale=beta)
        super(Gamma, self).__init__()

        # set distribution name/params for __str__ and friends
        self._str = "Gamma(alpha=%.5f, beta=%.5f)"

    @property
    def params(self):
        return (self.alpha, self.beta)


# ########################################################################## #
# Below we have other distributions that are not a part of scipy.stats       #
# ########################################################################## #

#  ------------------ #
#  NormalInverseGamma #
#  ------------------ #

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
