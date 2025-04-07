import setuptools

with open("README.md", "r", encoding = "utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name = "face-crop-plus-perkjam",
    version = "1.1.0",
    author = "Sweeney Todd",
    author_email = "junkmesend@gmail.com",
    license = "MIT",
    description = f"Automatic face aligner and cropper with quality "
                  f"enhancement and attribute parsing.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    keywords = [
        "face",
        "python",
        "pytorch",
        "alignment",
        "cropping",
        "super resolution",
        "quality enhancement",
        "parsing",
        "grouping",
        "attributes",
        "mask",
        "segmentation",
        "celeba",
    ],
    install_requires = [
        "tqdm",
        "unidecode",
        "opencv-python",
        "torch",
        "torchvision",
    ],
    classifiers = [
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "face-crop-plus-perkjam=face_crop_plus_perkjam.__main__:main",
            "face_crop_plus_perkjam=face_crop_plus_perkjam.__main__:main",
            "fcpp=face_crop_plus_perkjam.__main__:main",
        ]
    },
    python_requires = ">=3.10"