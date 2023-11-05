from setuptools import find_packages, setup


def get_version() -> str:
    with open("protonet/version.py", "r") as foo:
        version = foo.read().split("=")[-1].replace("'", "").replace('"', "").strip()
    return version


__author__ = {"name": "Ömer BORHAN", "email": "borhano.f.42@gmail.com"}

# load long description
with open("README.md", "r") as foo:
    long_description = foo.read()

# load requirements
with open("requirements.txt", "r") as foo:
    requirements = foo.read().split("\n")


setup(
    # package name `pip install protonet`
    name="protonet",
    # package version `major.minor.patch`
    version=get_version(),
    # small description
    description="Implementation of Prototypical Networks for Few-shot Learning",
    # long description
    long_description=long_description,
    # content type of long description
    long_description_content_type="text/markdown",
    # source code url for this package
    url="https://github.com/borhanMorphy/prototypical-networks",
    # author of the repository
    author=__author__["name"],
    # author's email adress
    author_email=__author__["email"],
    # package license
    license="MIT",
    # package root directory
    packages=find_packages(),
    # requirements
    install_requires=requirements,
    include_package_data=True,
    # keywords that resemble this package
    keywords=["lightning", "few shot learning", "meta learning"],
    zip_safe=False,
    # classifiers for the package
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)