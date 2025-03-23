from setuptools import setup, find_packages

setup(
    name="inference_opt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.19.5",
        "einops>=0.4.1",
        "tqdm",
        "transformers>=4.25.0"
    ],
    author="Mann Patel",
    author_email="manncodes@gmail.com",
    description="Implementation of Radix Attention, Multi-head Latent Attention (MLA), and Speculative Decoding",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/manncodes/inference-opt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
