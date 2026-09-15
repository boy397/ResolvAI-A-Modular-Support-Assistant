import setuptools

setuptools.setup(
    name="resolvai",
    version="0.1.0",
    description="Grounded AI customer support agent: intent classification, "
                 "retrieval-grounded reply drafting, and escalation routing.",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
