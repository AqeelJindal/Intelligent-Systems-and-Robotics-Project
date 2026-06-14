from setuptools import find_packages, setup

package_name = 'ros2_project_sc23aj2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cscajb',
    maintainer_email='x.wang16@leeds.ac.uk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main = ros2_project_sc23aj2.main:main',
            'colour_test = ros2_project_sc23aj2.colour_test:main',
            'nav_test = ros2_project_sc23aj2.nav_test:main'
        ],
    },
)
