from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    """This function will return a list of requirements"""
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements ]
        
        if HYPEN_E_DOT is requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
name='ML Project',
version='0.0.1',
author = 'Pravin Maurya',
author_email = 'PravinCoder@gmail.com',
packages=find_packages(),
install_requirements = get_requirements('requirements.txt'),
)
