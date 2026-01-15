from setuptools import setup, find_packages
import os

this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_desc = f.read()

requirements = [
    "groq>=0.10.0",
    "tiktoken>=0.7.0",
    "langchain>=0.2.0",
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
    "pymupdf>=1.23.0",
    "python-docx>=1.1.0",
    "httpx==0.26.0",
]

setup(
    name="groqchat-lib",
    version="0.2.1",
    author="Timoshenko Dmitriy",
    author_email="mittimash1997@yandex.ru",
    description="Modular Groq chat client with sessions, RAG, and full audit log",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/mittimash/groqchat_lib.git",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)