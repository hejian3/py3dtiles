# How to release

- make sure you've run `pip install -e .[pack]`
- clean previous builds: `rm dist/ -rf`
- edit the CHANGELOG.md. The best way is to start with commitizen for that:
```bash
cz changelog --incremental --unreleased-version v4.0.0
```
and then edit it to make it more user readable. Especially, the `BREAKING
CHANGE` needs to be reviewed carefully and often to be rewritten, including
migration guide for instance.
- edit the version in `py3dtiles/__init__.py`
- create a merge request with these changes
- once it is merged, create a tagged release on gitlab.
- wait for the execution of pages that will update the documentation
- publish on pypi:
```bash
# create a package in dist/ folder
python -m build
# check everything is ok (replace <version> by the version you've just built)
twine check dist/py3dtiles-<version>*
# check your pypirc for authentication
# upload it to pypi, eventually using --repository for selecting the right authent
twine upload dist/py3dtiles-<version>*
```

Check if the doc for the new version is published.
