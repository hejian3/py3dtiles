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
- edit the version in [\_\_init\_\_.py](py3dtiles/__init__.py)
- edit the version in [sonar-project.properties](sonar-project.properties) (field `sonar.projectVersion`)
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

## Case: a python version is newly supported or dropped

Update the following files:
- [setup.py](setup.py)
  - modify `python_requires`
  - add or remove the python version in `classifiers` list
- [Dockerfile](Dockerfile)
  - check if the version of python is still supported by py3dtiles
  - if the python version in the Dockerfile is changed, regenerate [requirements.txt](requirements.txt). Be careful not to add unnecessary packages
- [sonar-project.properties](sonar-project.properties)
  - edit the `sonar.python.version` variable
- [.pre-commit-config.yaml](.pre-commit-config.yaml)
  - only in case of a python version that is no longer supported
  - change the `args` value for the hook `pyupgrade`
  - change the `args` value for the hook `black`
- [.gitlab-ci.yml](.gitlab-ci.yml)
  - in case of a new python version support
    - change the python docker image version of the jobs
    - add the new version in the python version matrix for the `test` job
  - in case of a python version that is no longer supported
    - remove the old version in the python version matrix for the `test` job

For the commit message, don't use `chore`, else the breaking change won't be displayed by commitizen...
