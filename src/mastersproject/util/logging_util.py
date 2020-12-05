# --- LOGGING UTIL ---
import functools
import logging
import time

default_logger = logging.getLogger("GTS.OVERWRITE_ME")


def timer(logger=default_logger, level="INFO", show_time=False):
    """Credits:
    https://realpython.com/primer-on-python-decorators/#decorators-with-arguments
    """

    def decorator_timer(func):
        """ Print the runtime of the decorated function"""

        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            lvl = logging.getLevelName(level)
            if not isinstance(lvl, int):
                lvl = logging.getLevelName("INFO")
                logging.warning("Unrecognised logging level!")
            ctime = f"({time.asctime()})" if show_time else ""
            logger.log(level=lvl, msg=f"Calling {func.__name__} {ctime}")
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            logger.log(
                level=lvl, msg=f"Finished {func.__name__!r} in {run_time:.4f} secs"
            )
            return value

        return wrapper_timer

    return decorator_timer


def trace(logger=default_logger, timeit=True, level="INFO"):
    """Credits:
    https://realpython.com/primer-on-python-decorators/#decorators-with-arguments
    """

    def decorator_trace(func):
        """Print the function signature and return value.
        Optionally print the runtime (default: True)
        """

        @functools.wraps(func)
        def wrapper_debug(*args, **kwargs):
            lvl = logging.getLevelName(level)
            if not isinstance(lvl, int):
                lvl = logging.getLevelName("INFO")
                logging.warning("Unrecognised logging level!")
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            logger.log(level=lvl, msg=f"Calling {func.__name__}({signature})")
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            if timeit:
                logger.log(
                    level=lvl, msg=f"Finished {func.__name__!r} in {run_time:.4f} secs"
                )
            logger.log(level=lvl, msg=f"{func.__name__!r} returned {value!r}")
            return value

        return wrapper_debug

    return decorator_trace


def __setup_logging(path, log_fname="results.log"):
    # reset root logger
    logging.getLogger().handlers = []
    path = str(path)
    # GTS logger
    gts_logger = logging.getLogger("GTS")
    gts_logger.setLevel(logging.INFO)
    if gts_logger.handlers:
        gts_logger.handlers = []

    # PorePy logger
    pp_logger = logging.getLogger("porepy")
    pp_logger.setLevel(logging.DEBUG)
    if pp_logger.handlers:
        pp_logger.handlers = []

    common_format = logging.Formatter(logging.BASIC_FORMAT)

    # Add handler for logging debug messages to file.
    fh = logging.FileHandler(path + "/" + log_fname, "w+")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(common_format)

    gts_logger.addHandler(fh)
    pp_logger.addHandler(fh)

    # Add handler for logging info messages to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(common_format)

    gts_logger.addHandler(ch)
    pp_logger.addHandler(ch)
