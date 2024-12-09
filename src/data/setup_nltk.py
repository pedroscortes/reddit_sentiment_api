# src/data/setup_nltk.py

import nltk
import ssl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_nltk_resources():
    """Download all required NLTK resources"""
    try:
        # Create an SSL context to handle download issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Resources we need
        resources = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "omw-1.4"]  # Open Multilingual Wordnet

        for resource in resources:
            logger.info(f"Downloading {resource}...")
            nltk.download(resource)

        logger.info("All NLTK resources downloaded successfully!")

    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {str(e)}")
        raise


if __name__ == "__main__":
    download_nltk_resources()
