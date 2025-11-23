from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop


PROJECT_ROOT = Path(__file__).parent.resolve()
VENDOR_DIR = PROJECT_ROOT / "distribution_shift_perception" / "torch-two-sample"
VENDOR_PACKAGE = "torch_two_sample"
VENDOR_PACKAGE_PATH = VENDOR_DIR / VENDOR_PACKAGE


def build_vendor_extensions() -> None:
    if not VENDOR_PACKAGE_PATH.exists():
        return

    subprocess.check_call(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=str(VENDOR_DIR),
    )


class build_py(_build_py):
    def run(self) -> None:  # type: ignore[override]
        build_vendor_extensions()
        super().run()


class develop(_develop):
    def run(self) -> None:  # type: ignore[override]
        build_vendor_extensions()
        super().run()


packages = find_packages()
if VENDOR_PACKAGE not in packages:
    packages.append(VENDOR_PACKAGE)

package_dir = {"": ".", VENDOR_PACKAGE: str(VENDOR_PACKAGE_PATH)}
package_data = {VENDOR_PACKAGE: ["*.so", "*.pyd", "*.dll"]}

setup(
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    include_package_data=True,
    cmdclass={"build_py": build_py, "develop": develop},
)
