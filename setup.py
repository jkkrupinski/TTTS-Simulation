from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="tttssim",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements("requirements.txt"),
    extras_require={},
    author="Jakub Krupiński",
    author_email="jakub.k.krupinski@gmail.com",
    description="TTTS robot simulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jkkrupinski/TTTS-Simulation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires=">=3.6",
)
