import distcan as dc
import numpy as np
from numpy.testing import assert_allclose
import scipy as sp
import scipy.stats as st


# Get some random places to check pdf
np.random.seed(1234)
x = np.random.rand(10)

# Create chi distributions
chi_dc = dc.univariate.Chi(5)
chi_sp = st.chi(5)

# Check chi cdfs/pdfs against each other
chidc_cdf = chi_dc.cdf(x)
chisp_cdf = chi_sp.cdf(x)
assert_allclose(chidc_cdf, chisp_cdf)

# Create chi2 distributions
chi2_dc = dc.univariate.Chisq(5)
chi2_sp = st.chi2(5)

# Check chi2 cdfs/pdfs against each other
chi2dc_cdf = chi2_dc.cdf(x)
chi2sp_cdf = chi2_sp.cdf(x)
assert_allclose(chi2dc_cdf, chi2sp_cdf)

# Create Uniform
un_dc = dc.univariate.Uniform(0., 7.5)
un_st = st.uniform(loc=0., scale=7.5)

# Check uniform cdf/pdfs against each other
assert_allclose(un_dc.pdf(x), un_st.pdf(x))
assert_allclose(un_dc.cdf(x), un_st.cdf(x))