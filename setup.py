from setuptools import setup, find_packages

setup(name='sarnet-td3',
      version='0.0.1',
      description='SARNET',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym==0.10.5', 'numpy-stl', 'gast==0.2.2']
)
