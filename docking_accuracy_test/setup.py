from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'docking_accuracy_test'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jeongmin',
    maintainer_email='jeongmin@todo.todo',
    description='Nav2 docking accuracy measurement package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'docking_accuracy_test_node = docking_accuracy_test.docking_accuracy_test_node:main',
            'gt_localization_node = docking_accuracy_test.gt_localization_node:main',
            'random_staging_docking_test_node = docking_accuracy_test.random_staging_docking_test_node:main',
        ],
    },
)
