import sys
import logging


LOGGER_NAME = "adversarial"


def setup_custom_logger(log_file="log/output.log", logger_name=None):
    """
    Set up a logger object to print info to stdout and debug to file.
    """
    logger_name = logger_name or LOGGER_NAME
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="[%(asctime)s - %(levelname)-4s] >> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.FileHandler(log_file, mode="a")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
