"""One exception per failure mode the CLI maps onto a distinct exit code."""


class LacError(Exception):
    """Base class. Anything raised deliberately by this package is a subclass."""


class ConfigError(LacError):
    """The comparison was not runnable as specified: bad flags, bad alpha, unknown run."""


class DataError(LacError):
    """A label set could not be read, or two label sets do not describe the same items."""


class UnderpoweredError(LacError):
    """The labelled sample is too small to support the guarantee the caller asked for."""
