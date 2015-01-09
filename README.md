# distcan

Probability **dist**ributions for python in their **can**onical form. Documentation (TODO: link)

`scipy.stats` is the go to library for working with probability distributions in python. It is an impressive package that exposes a consistent API for working with almost 100 distributions. But, there are some shortcomings...


TODO: pull from blog post to talk about problems I have with `scipy.stats`

## Enter `distcan`

The `distcan` library aims to address these problems in an easily extensible way. Some goals of this project are

TODO: pull goals section from blog post

By leveraging the great code in `scipy.stats`, we are well on our way to completing these goals.

### Adding a new distribution

Adding a new distribution, based on one in `scipy.stats`, is extremely easy. To see just how easy it is, we will consider an example and then walk through how it works. Below is the actual implementation of the InverseGamma distribution from `distcan` (as of 2015-01-08):

```python
class InverseGamma(CanDistFromScipy):                                       # 1

    _metadata = {
        "pdf_tex": (r"p(x;\alpha,\beta)=\frac{\beta^{\alpha}}{\Gamma(\alpha)}"
                    + r"x^{-\alpha-1}\exp\left(-\frac{\beta}{x}\right)"),

        "cdf_tex": r"\frac{\Gamma(\alpha, \beta / x)}{\Gamma(\alpha)}",

        "param_names": ["alpha", "beta"],

        "param_descrs": ["Shape parameter (must be >0)",
                         "Scale Parameter (must be >0)"],

        "_str": "InverseGamma(alpha=%.5f, beta=%.5f)"}                      # 2

    # set docstring
    __doc__ = _create_class_docstr("InverseGamma", **_metadata)             # 3

    def __init__(self, alpha, beta):                                        # 4
        self.alpha = alpha                                                  # 5
        self.beta = beta

        # set dist before calling super's __init__
        self.dist = st.invgamma(alpha, scale=beta)                          # 6
        super(InverseGamma, self).__init__()                                # 7

    @property                                                               # 8
    def params(self):
        return (self.alpha, self.beta)
```

I have labeled certain lines of the code above. Let's analyze what is happening item by item:

1. Notice that we are subclassing `CanDistFromScipy`. This is a class defined in `distcan.scipy_wrap` that does almost all the work for us, including defining methods and setting docstrings for each method.
2. `_metadata` is a dict that is used to set the docstring of the class and  each method as well as the `__str__` and `__repr__` methods. For an explanation of what this dict should contain and how it is used, see the [metadata section](TODO: link) of the docs
3. This line uses the information from the `_metadata` dict to set the docstring for the class
4. The arguments to `__init__` method are the canonical parameters and associated names for the distribution
5. These arguments are stored as attributes of the class
6. Create an internal instance of the distribution, based on the implementation in `scipy.stats`. This is where we map canonical parameter names into the notation from `scipy.stats`
7. Call the `__init__` method of `CanDistFromScipy`. This is where the heavy lifting happens
8. Set an additional property called `params` that returns the parameters of the distribution
