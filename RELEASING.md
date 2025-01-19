1. Update the version in `src/quantity_array/__init__.py`
2. Update the lockfile with `pixi update && pixi install`
3. Push your changes
4. Cut a release via the GitHub GUI
5. Update the version to `{next micro version}.dev0`
6. `pixi clean cache --pypi && pixi update && pixi install`
7. Push your changes
8. Merge the automated PR to conda-forge/quantity-array-feedstock
