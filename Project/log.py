import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - {%(filename)s:%(lineno)d} - %(levelname)s - %(message)s')
handler = logging.FileHandler('PHYS389.log', mode='w+')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)