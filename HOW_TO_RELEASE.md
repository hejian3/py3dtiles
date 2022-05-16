# How to release

- edit the CHANGELOG.md
- edit the version in `py3dtiles/__init__.py`
- create a tagged release on gitlab. 
- wait for pages to execute and update the documentation
- publish on pypi:
```bash
# create a package in dist/ folder
python setup.py sdist
# check your pypirc for authentication
# upload it to pypi, eventually using --repository for selecting the right authent
python setup.py sdist upload
```

Check if the doc for the new version is published.
