import subprocess
import sys
import logging
from typing import List, Tuple
import importlib.metadata


def setup_logging():
    """Configure logging for the installation process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/installation.log')
        ]
    )
    return logging.getLogger(__name__)


def parse_requirements(filename: str) -> List[Tuple[str, str]]:
    """
    Parse requirements.txt and return list of (package_name, version) tuples
    Ignores comments and empty lines
    """
    requirements = []
    with open(filename, 'r') as file:
        for line in file:
            # Remove comments if they exist on the same line as a package
            if '#' in line:
                line = line[:line.index('#')]

            line = line.strip()
            if line and not line.startswith('#'):
                # Handle version specifiers
                if '==' in line:
                    package, version = line.split('==')
                    requirements.append((package.strip(), version.strip()))
                else:
                    requirements.append((line, ''))
    return requirements


def is_package_installed(package_name: str, required_version: str = '') -> bool:
    """Check if a package is installed with the required version"""
    try:
        installed_version = importlib.metadata.version(package_name)
        if required_version:
            return installed_version == required_version
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def install_package(package_name: str, version: str, logger) -> bool:
    """Install a package using pip"""
    try:
        package_spec = f"{package_name}=={version}" if version else package_name
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_spec])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {str(e)}")
        return False


def setup_nltk(logger):
    """Download required NLTK data"""
    try:
        import nltk
        nltk_packages = ['punkt', 'stopwords', 'wordnet']
        for package in nltk_packages:
            logger.info(f"Downloading NLTK package: {package}")
            nltk.download(package, quiet=True)
        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK packages: {str(e)}")
        return False


def setup_spacy(logger):
    """Download SpaCy English language model"""
    try:
        subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download SpaCy model: {str(e)}")
        return False


def check_and_install_dependencies():
    """Main function to check and install all dependencies"""
    logger = setup_logging()
    logger.info("Starting dependency check and installation...")

    # Parse requirements.txt
    try:
        requirements = parse_requirements('requirements.txt')
    except FileNotFoundError:
        logger.error("requirements.txt not found!")
        return False

    # Check and install each package
    all_successful = True
    for package_name, version in requirements:
        if not is_package_installed(package_name, version):
            logger.info(f"Installing {package_name} {'version ' + version if version else ''}")
            if not install_package(package_name, version, logger):
                all_successful = False
        else:
            logger.info(f"{package_name} {'version ' + version if version else ''} is already installed")

    # Setup NLTK
    if all_successful:
        logger.info("Setting up NLTK...")
        if not setup_nltk(logger):
            all_successful = False

    # Setup SpaCy
    if all_successful:
        logger.info("Setting up SpaCy...")
        if not setup_spacy(logger):
            all_successful = False

    if all_successful:
        logger.info("All dependencies installed successfully!")
    else:
        logger.error("Some installations failed. Check the logs for details.")

    return all_successful


if __name__ == "__main__":
    success = check_and_install_dependencies()
    sys.exit(0 if success else 1)
