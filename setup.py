from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = "-e ."


def get_requirements(req_file: str) -> List[str]:
    """
    Load requirements from a .txt file and return them as a list of strings.
    """
    try:
        with open(req_file, "r") as file:
            requirements = [line.strip() for line in file]

            if HYPEN_E_DOT in requirements:
                requirements.remove(HYPEN_E_DOT)
        return requirements
    except FileNotFoundError:
        print(f"Error: File '{req_file}' not found.")
        return []
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return []


setup(
    name="GenericMLProject",
    version="0.0.1",
    author="Smaran",
    author_email="smarandhg@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
