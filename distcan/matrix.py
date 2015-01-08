"""
Matrix variate distributions

@author : Spencer Lyon <spencer.lyon@stern.nyu.edu>
@date : 2015-01-07

"""
from math import log, sqrt, pi
import numpy as np
import scipy.stats as st
import scipy.linalg as la
from scipy.special import gammaln, digamma

__all__ = ["Wishart", "InverseWishart"]


# imports to match julia
def logdet(x):
    return np.linalg.slogdet(x)[1]

logtwo = log(2.0)


def _unwhiten_cf(cf, x):
    """
    Unwhiten a matrix

    Parameters
    ----------
    cf : array_like(ndim=2, dtype=float)
        The upper triangular part of the cholesky decomposition of a

    x : array_like(ndim=2, dtype=float)
        Not sure, just blindly porting...

    Returns
    -------
    updated matrix.

    """
    return cf.T.dot(x)


def unwhiten(a, x):
    """
    Unwhiten a matrix

    Parameters
    ----------
    a : array_like(ndim=2, dtype=float)
        Not sure, just blindly porting...

    x : array_like(ndim=2, dtype=float)
        Not sure, just blindly porting...

    Returns
    -------
    updated matrix.

    """
    return _unwhiten_cf(la.cholesky(a), x)


def isposdef(X):
    "Return if matrix is positive definite. Relies on cholesky decomp"
    try:
        la.cholesky(X)  # will raise LinAlgError if not positive def
        return True
    except la.LinAlgError:
        return False


#  ---------------- #
#  utility routines #
#  ---------------- #

# Multidimensional gamma / partial gamma function
def lpgamma(p, a):
    """
    Multidimensional gamma / partial gamma function

    Parameters
    ----------
    p : int
        something....

    a : float
        something....

    Returns
    -------
    Multidimensional gamma / partial gamma function

    """
    res = p * (p - 1.0) / 4.0 * log(pi)
    for ii in range(1, p+1):
        res += gammaln(a + (1.0 - ii) / 2.0)

    return res


class WishartIWishartParent(object):

    def __init__(self, df, S):
        p = S.shape[0]
        self._p = p

        assert df > p - 1, "df should be greater than S.shape[0] - 1."

        self.df = df
        self.S = S
        self.c0 = self._c0()

        self._S_cf = la.cholesky(S)  # the triu part of cholesk decomp

    def __str__(self):
        nm, df, S = self._dist_name, self.df, self.S
        return "%s\n  -df: %i\n  -S:\n%s" % (nm, df, S)

    def __repr__(self):
        return self.__str__()

    def insupport(self, X):
        """
        Test if a matrix X is in the support of the distribution.
        Returns true iff the X is positive definite

        Parameters
        ----------
        X : array_like (dtype=float, ndim=2)
            A test matrix

        Returns
        -------
        ans : bool
            A boolean indicating if the matrix is in the support of the
            distribution

        """
        return isposdef(X)

    def logpdf(self, X):
        if X.ndim == 2:  # single point
            return self._logpdf1(X)
        else:
            # make sure we have proper dimensions
            (n, p1, p2) = X.shape

            if p1 != p2 or p1 != self._p:
                msg = "Incorrect dimensions for logpdf a multiple points."
                msg += "Must have dimensions (n, p, p) - n is # of points"
                raise ValueError(msg)

            out = np.empty(n)

            for i in range(n):
                out[i] = self._logpdf1(X[i])

    def pdf(self, X):
        """
        Evaluate the pdf of the distribution at various points

        Parameters
        ----------
        X : array_like(dtype=float, ndim=(2,3))
            Where to evaluate the pdf. If 2 dimensional, evaluate at
            single point. If X is three dimensional

        Returns
        -------
        out : scalar_or_array(dtype=float, ndim=(0,1))

        """
        return np.exp(self.logpdf(X))

    def rand(self, n=1):
        """
        Generate random samples from the distribution

        Parameters
        ----------
        n : int, optional(default=1)
            The number of samples to generate

        Returns
        -------
        out : array_like
            The generated samples

        """
        if n == 1:
            return self._rand1()
        else:
            out = np.empty((n, self._p, self._p))
            for i in range(n):
                out[i] = self._rand1()

            return out


#  --------------------  #
#  Wishart Distribution  #
#  --------------------  #

class Wishart(WishartIWishartParent):
    """
    Wishart distribution

    Parameters
    ----------
    df : int
        The degrees of freedom parameter. Must be a positive integer

    S : array_like(dtype=float, ndim=2)
        The scale matrix

    Notes
    -----
    Follows the wikipedia parameterization.

    Translation of the associated file from Distributions.jl

    """
    def __init__(self, df, S):
        super(Wishart, self).__init__(df, S)
        self._dist_name = "Wishart Distribution"

    def _c0(self):
        "the logarithm of normalizing constant in pdf"
        h_df = self.df / 2
        p, S = self._p, self.S

        return h_df * (logdet(S) + p * logtwo) + lpgamma(p, h_df)

    def _genA(self):
        """
        Generate the matrix A in the Bartlett decomposition

          A is a lower triangular matrix, with

              A(i, j) ~ sqrt of Chisq(df - i + 1) when i == j
                      ~ Normal()                  when i > j
        """
        p, df = self._p, self.df
        A = np.zeros((p, p))

        for i in range(p):
            A[i, i] = sqrt(st.chi2.rvs(df - i))

        for j in range(p-1):
            for i in range(j+1, p):
                A[i, j] = np.random.randn()

        return A

    def _rand1(self):
        "generate a single random sample"
        Z = _unwhiten_cf(self._S_cf, self._genA())
        return Z.dot(Z.T)

    @property
    def mean(self):
        return self.df * self.S

    @property
    def mode(self):
        r = self.df - self._p - 1.0
        if r > 0.0:
            return self.S * r
        else:
            raise ValueError("mode is only defined when df > p + 1")

    @property
    def meanlogdet(self):
        p, df, S = self._p, self.df, self.S
        v = logdet(S) + p * logtwo

        v += digamma(0.5 * (df - (np.arange(p)))).sum()

        return v

    @property
    def entropy(self):
        p, df, c0 = self._p, self.df, self.c0
        return c0 - 0.5*(df - p - 1) * self.meanlogdet + 0.5*df*p

    def _logpdf1(self, X):
        p, df, S, c0 = self._p, self.df, self.S, self.c0
        Xcf = la.cholesky(X)

        # multiply logdet by 2 b/c julia does in logdet(::CholFact)
        return 0.5*((df - (p + 1))*2*logdet(Xcf) -
                    np.trace(la.solve(S,  X))) - c0


#  -------------- #
#  InverseWishart #
#  -------------- #

class InverseWishart(WishartIWishartParent):
    """
    Inverse Wishart distribution

    Parameters
    ----------
    df : int
        The degrees of freedom parameter. Must be a positive integer

    S : array_like(dtype=float, ndim=2)
        The scale matrix

    Notes
    -----
    Follows the wikipedia parameterization.

    Translation of the associated file from Distributions.jl.

    NOTATION: I changed Psi to S

    """

    def __init__(self, df, S):
        super(InverseWishart, self).__init__(df, S)
        self._dist_name = "Inverse Wishart Distribution"
        self._Wishart = Wishart(df, la.inv(S))

    def _c0(self):
        "the logarithm of normalizing constant in pdf"
        h_df = self.df / 2
        p, S = self._p, self.S

        return h_df * (p * logtwo - logdet(S)) + lpgamma(p, h_df)

    @property
    def mean(self):
        df, p, S = self.df, self._p, self.S
        r = df - (p + 1)
        if r > 0.0:
            return S * (1.0 / r)
        else:
            raise ValueError("mean only defined for df > p + 1")

    @property
    def mode(self):
        S, df, p = self.S, self.df, self._p
        return S / (df + p + 1.0)

    def _logpdf1(self, X):
        p, df, S, c0 = self._p, self.df, self.S, self.c0
        Xcf = la.cholesky(X)

        # we use the fact: trace(S * inv(X)) = trace(inv(X) * S) = trace(X\S)
        return -0.5*((df + p + 1)*2*logdet(Xcf) +
                     np.trace(la.solve(Xcf, S))) - c0

    def _rand1(self):
        return la.inv(self._Wishart._rand1())
