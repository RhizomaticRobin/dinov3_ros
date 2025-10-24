from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    # Get the directory where this setup.py is located
    here = os.path.abspath(os.path.dirname(__file__))
    req_path = os.path.join(here, filename)

    # If requirements.txt doesn't exist, return empty list
    if not os.path.exists(req_path):
        return []

    with open(req_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='dinov3_ros',
    version='0.1.0',
    packages=find_packages(include=['dinov3_toolkit', 'dinov3_toolkit.*']),
    install_requires=parse_requirements('requirements.txt'),
    # package_data={
    #     "dinov3_toolkit": [
    #         "heads/*/weights/*.pth",
    #         "backbones/weights/*.pth"
    #     ]
    # },
    include_package_data=True,
)