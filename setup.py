from setuptools import setup, find_packages

setup(
    name='EXPERIMENTAL_DESING',
    version='0.1.0',
    description='Herramientas para diseño experimental y análisis estadístico',
    author='Alexis',
    author_email='Ext_ASucasacaV@alicorp.com.pe',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
    ],
    python_requires='>=3.7',
)


