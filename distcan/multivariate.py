"""
Multivariate distributions

@author : Spencer Lyon <spencer.lyon@stern.nyu.edu>
@date : 2015-01-07

"""
import scipy.stats as st

__all__ = ["MvNormal", "MultivariateNormal"]


#  -------- #
#  MvNormal #
#  -------- #

# NOTE: can't subclass CanDistFromScipy because not all methods are
#       implemented within scipy
class MvNormal(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self._str = "MvNormal(mu=%s, sigma=%s)"
        self.dist = st.multivariate_normal(mean=mu, cov=sigma)

        self.dim = self.dist.dim
        self.pdf = self.dist.pdf
        self.logpdf = self.dist.logpdf
        self.prec_U = self.dist.prec_U

    def __len__(self):
        return self.dim

    def __str__(self):
        return self._str % self.params

    def __repr__(self):
        return self.__str__()

    @property
    def params(self):
        return (self.mu, self.sigma)

    @property
    def mean(self):
        return self.mu

    @property
    def cov(self):
        return self.sigma

    @property
    def entropy(self):
        return self.dist.entropy()

    def rand(self, *args):
        return self.dist.rvs(size=args)


MultivariateNormal = MvNormal
