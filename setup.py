from setuptools import setup

setup(name='php-webshell-classifier',
      version='0.1',
      description='PHP Webshell Classifier based on NeoPI and Yara Rules',
      url='https://github.com/bmoss6/ml_478_php_webshell',
      author='Blake Moss',
      author_email='blake_moss@byu.edu',
      license='MIT',
      packages=['predictor', 'trainer', 'feature_extractor'],
      install_requires=[
          'yara',
          'sklearn'
      ],
      zip_safe=False)
