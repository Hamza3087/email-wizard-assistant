from setuptools import setup, find_packages

setup(
    name="email-wizard-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "transformers>=4.15.0",
        "sentence-transformers>=2.2.0",
        "flask>=2.0.0",
        "faiss-cpu>=1.7.0",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An Email Wizard Assistant using RAG technology",
    keywords="email, RAG, NLP, AI",
    url="https://github.com/yourusername/email-wizard-assistant",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)
