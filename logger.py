import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import datetime

def get_logger(name: str):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger
    
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents= True, exist_ok= True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    log_file = log_dir/ f"app_{timestamp}.log"

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        filename= log_file,
        maxBytes= 10*1024*1024,
        backupCount= 5,
        encoding= 'utf-8'
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger