#!/usr/bin/env python3
from typing import List
import os
from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))

PLUGIN_ENTRY_POINT = (
    "alena-plugin-hotword-vosk = alena_plugin_hotword_vosk:VoskHotword"
)


def required(requirements_file: str) -> List[str]:
    """Возвращает список зависимостей, исключая комментированные строки."""
    with open(os.path.join(BASEDIR, requirements_file), "r") as f:
        requirements = f.read().splitlines()
        return [
            pkg
            for pkg in requirements
            if pkg.strip() and not pkg.startswith("#")
        ]


def get_version() -> str:
    """Возвращает информацию о версии плагина."""
    version = None
    version_file = os.path.join(
        BASEDIR, "alena_plugin_hotword_vosk", "version.py"
    )
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if "VERSION_MAJOR" in line:
                major = line.split("=")[1].strip()
            elif "VERSION_MINOR" in line:
                minor = line.split("=")[1].strip()
            elif "VERSION_BUILD" in line:
                build = line.split("=")[1].strip()
            elif "VERSION_ALPHA" in line:
                alpha = line.split("=")[1].strip()

            if (
                major and minor and build and alpha
            ) or "# END_VERSION_BLOCK" in line:
                break
    version = f"{major}.{minor}.{build}"
    if alpha and int(alpha) > 0:
        version += f"a{alpha}"
    return version


setup(
    name="alena-plugin-hotword-vosk",
    version=get_version(),
    description="Kaldi wake word plugin for Alena",
    url="https://github.com/abelinsky/alena-templates",
    author="Alexander Belinsky",
    author_email="belinskyab@mail.ru",
    license="Apache-2.0",
    packages=["alena_plugin_hotword_vosk"],
    install_requires=required("requirements.txt"),
    zip_safe=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="alena plugin wake word",
    entry_points={"alena.plugin.wake_word": PLUGIN_ENTRY_POINT},
)
