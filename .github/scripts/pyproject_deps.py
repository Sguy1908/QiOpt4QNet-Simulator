"""Print the dependencies declared in QNet_Sim/pyproject.toml, one per line.

The project has no importable top-level package, so ``pip install .[all]``
cannot be used in CI. This keeps pyproject.toml the single source of truth:
runtime dependencies plus the optional-dependency groups named on the command
line (default: ``all``).
"""

import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

with open("QNet_Sim/pyproject.toml", "rb") as f:
    project = tomllib.load(f)["project"]

groups = sys.argv[1:] or ["all"]
deps = list(project["dependencies"])
for group in groups:
    deps += project["optional-dependencies"][group]

seen = set()
for dep in deps:
    if dep not in seen:
        seen.add(dep)
        print(dep)
