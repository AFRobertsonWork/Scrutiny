from setuptools import find_packages, setup

package_name = 'scrutiny_sleek_gui'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),

	data_files=[
	    ('share/ament_index/resource_index/packages', ['resource/scrutiny_sleek_gui']),
	    ('share/scrutiny_sleek_gui', ['package.xml']),
	    ('share/scrutiny_sleek_gui/launch', ['launch/scrutiny_system.launch.py']),
	],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'scrutiny_gui = scrutiny_sleek_gui.gui_node:main',
        ],
    },
)
