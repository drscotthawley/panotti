from setuptools import setup
from setuptools import find_packages


setup(name='Panotti',
      version='1.0.0',
      description='Multi-Channel Audio Classifier',
      author='Scott Hawley',
      author_email='scott.hawley@belmont.edu',
      url='https://github.com/drscotthawley/panotti',
      download_url='https://github.com/drscotthawley/panotti/tarball/1.0.0',
      license='MIT',
      install_requires=['keras', 'librosa', 'h5py'],
      extras_require={
          'headgames': ['pygame'],
      },
      packages=find_packages()) 
