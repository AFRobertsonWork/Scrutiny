from setuptools import find_packages, setup

package_name = "scrutiny_measurement"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aditya Jadhav",
    maintainer_email="aditya@example.com",
    description="RGB and depth-based object measurement for Scrutiny.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "measurement_node = "
            "scrutiny_measurement.measurement_node:main",
        ],
    },
)
