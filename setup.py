#!/usr/bin/env python
"""
Setup script for OpenWearables

This script handles the installation and distribution of the OpenWearables
AI-powered wearable health monitoring platform.
"""

import os
import re
from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Get version string from __init__.py
def get_version():
    version_file = this_directory / "openwearables" / "__init__.py"
    version_content = version_file.read_text(encoding='utf-8')
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Read requirements from files
def read_requirements(filename):
    requirements_file = this_directory / filename
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and requirement file references
                if line and not line.startswith('#') and not line.startswith('-r'):
                    requirements.append(line)
            return requirements
    return []

# Core requirements
install_requires = read_requirements('requirements.txt')

# Optional dependencies - ensure all are lists of strings
extras_require = {
    'mlx': [
        'mlx>=0.0.4',
    ],
    'cuda': [
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'torchaudio>=2.0.0',
    ],
    'llm': [
        'langchain>=0.0.300',
        'langchain-community>=0.0.20',
        'transformers>=4.30.0',
        'accelerate>=0.20.0',
        'bitsandbytes>=0.39.0',
    ],
    'test': [
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'pytest-asyncio>=0.21.0',
        'pytest-benchmark>=4.0.0',
        'pytest-mock>=3.11.0',
        'hypothesis>=6.82.0',
        'factory-boy>=3.3.0',
    ],
    'docs': [
        'sphinx>=7.1.0',
        'sphinx-rtd-theme>=1.3.0',
        'sphinx-autodoc-typehints>=1.24.0',
        'myst-parser>=2.0.0',
    ],
    'deploy': [
        'docker>=6.1.0',
        'kubernetes>=27.2.0',
        'gunicorn>=21.2.0',
        'celery>=5.3.0',
        'redis>=4.6.0',
    ],
}

# Only add dev requirements if we can successfully read them
dev_requirements = read_requirements('requirements-dev.txt')
if dev_requirements:
    extras_require['dev'] = dev_requirements

# Add 'all' option that includes most extras (excluding dev)
extras_require['all'] = list(set(
    extras_require['mlx'] + 
    extras_require['cuda'] + 
    extras_require['llm'] + 
    extras_require['deploy']
))

# Package data
package_data = {
    'openwearables': [
        'ui/templates/*.html',
        'ui/static/css/*.css',
        'ui/static/js/*.js',
        'ui/static/img/*',
        'config/*.json',
        'models/configs/*.json',
        'data/schemas/*.json',
    ]
}

# Data files
data_files = [
    ('share/openwearables/config', ['config/default.json'] if os.path.exists('config/default.json') else []),
    ('share/openwearables/docs', ['README.md', 'LICENSE'] if os.path.exists('LICENSE') else ['README.md']),
]

# Entry points
entry_points = {
    'console_scripts': [
        'openwearables=openwearables.cli.openwearables_cli:main',
    ],
}

# Setup configuration
setup(
    name="openwearables",
    version=get_version(),
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="Open-source platform for real-time wearable health monitoring and analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openwearables/openwearables",
    project_urls={
        "Homepage": "https://github.com/openwearables/openwearables",
        "Documentation": "https://docs.openwearables.org",
        "Repository": "https://github.com/openwearables/openwearables.git",
        "Bug Reports": "https://github.com/openwearables/openwearables/issues",
        "Changelog": "https://github.com/openwearables/openwearables/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    package_data=package_data,
    data_files=data_files,
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
        "Topic :: Internet :: WWW/HTTP :: WSGI :: Application",
    ],
    keywords=[
        "wearables", "health", "monitoring", "ai", "machine-learning", 
        "sensors", "ecg", "ppg", "mlx", "cuda", "privacy", "healthcare",
        "biometrics", "real-time", "analytics", "dashboard"
    ],
    zip_safe=False,
    platforms=["any"],
    license="MIT",
    
    # Additional metadata
    maintainer="OpenWearables Team",
    maintainer_email="dev@openwearables.org",
    
    # Custom commands for development
    cmdclass={},
    
    # Package configuration
    options={
        'bdist_wheel': {
            'universal': False,  # Not universal since we have platform-specific optimizations
        },
    },
) 