from setuptools import setup, find_packages

setup(
    name="Orange3-CustomAddon",
    version="0.1",
    description="Custom Orange3 add-on with Preprocessing, Autoencoder, and Classifier widgets",
    packages=find_packages(),
    package_data={
        'orangecontrib.customaddon.widgets': ['widget_icons/*.svg'],
    },
    install_requires=[
        'Orange3',
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'matplotlib',
        'seaborn'
    ],
    entry_points={
        'orange.widgets': (
            'CustomAddon = orangecontrib.customaddon.widgets',
        ),
        'orange.canvas.help': (
            'html-index = orangecontrib.customaddon.widgets:WIDGET_HELP_PATH',
        )
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
