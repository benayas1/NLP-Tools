import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(where=".")

setuptools.setup(
    name="nlp",  # Replace with your own username
    version="0.0.1",
    author="EPAM",
    author_email="alberto_jose_benayas@epam.com",
    description="A small nlp project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url=gitlab project,
    classifiers=[
                "Programming Language :: Python :: 3",
                # "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                ],
    python_requires='>=3.6',
    package_dir={"": "."},
    packages=packages
)
