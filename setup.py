from distutils.core import setup
setup(
  name = 'NeuralNetworkSelection',
  packages = ['NeuralNetworkSelection'],
  version = '1.1.0',
  license='MIT',
  description = 'Module to select the best sklearn neural network',
  author = 'class PythonAddict',
  url = 'https://github.com/classPythonAddike/NeuralNetworkSelection',
  download_url = 'https://github.com/classPythonAddike/NeuralNetworkSelection/archive/v1_0_0.tar.gz',
  keywords = ['Selection', 'scikit-learn', 'Neural Networks'],
  install_requires=['scikit-learn'],
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
	'Programming Language :: Python :: 3.7',
	'Programming Language :: Python :: 3.8',
  ],
)