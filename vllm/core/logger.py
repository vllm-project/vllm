import logging
import os

# initialize logger
logger = logging.getLogger('vllm_logger')
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)

# clear the file
open('debug.log', 'w').close()

# only print to the file 
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(file_handler)

logger.debug('Logger initialized')