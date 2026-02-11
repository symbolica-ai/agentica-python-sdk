import re

from setuptools import setup


def version_scheme(version):
    """Custom version scheme for external/standalone repo builds.

    Uses git tag as semver directly.
    Tag format: v0.3.2 or v0.3.2.dev2
    """
    tag = str(version.tag)
    distance = version.distance or 0

    # Tag is the semver
    tag_match = re.match(r'^v?(\d+\.\d+\.\d+)(?:\.dev(\d+))?$', tag)
    if tag_match:
        base_version = tag_match.group(1)
        tag_dev = int(tag_match.group(2)) if tag_match.group(2) else 0
        dev_num = tag_dev + distance

        if dev_num == 0:
            return base_version
        return f"{base_version}.dev{dev_num}"

    return "0.0.0.dev0"


setup(
    use_scm_version={
        "version_scheme": version_scheme,
        "local_scheme": "no-local-version",
    }
)
