# How to contribute to py3dtiles ?

## Report a bug

If you think you've found a bug in py3dtiles, first search the py3dtiles issues. If an issue already exists, you can add a comment with any additional information. Use reactions (not comments) to express your interest. This helps prioritizing issues.

If there is no related issue, submit a new one. Please include the following information, if relevant:
- Sample data to reproduce the issue
- Screenshot of the generated tileset if appropriate. Tileset can be visualized with tools like [giro3d](https://giro3d.org) or other 3dtiles viewers.
- The type and version of the OS and the version of python used.
- The exact version of py3dtiles. Did this work in a previous version or a next one?
- Add the tag ~Bug.

Ideas for how to fix or workaround the issue are welcome. Also mention if you are willing to help fixing it. If so, the py3dtiles team can often provide guidance and the issue may get fixed more quickly with your help.

## Suggest an improvement

If you think a feature should be added, it is useful to report the need.

As with bugs, check first if there is an issue with the same suggestion. If an issue already exists, you can add a comment with any additional information. Use reactions (not comments) to express your interest.
Else, you can create a new issue explaining the need and add the tag ~Feature.

## Participate in the development

We are open to any new contribution! We will try to give you a prompt feedback, review and merge your MR. To simplify the process, we invite you to read and follow these guidelines.

If you are making major changes to the code, you are encouraged to open an issue first to discuss the best way to integrate your code.

### CI jobs

Each MR will execute a CI pipeline. The CI will check:
 - the format of the commits messages
 - the syntax and the format of the code
 - the validity of all type annotations
 - that the tests pass with all versions of python supported
 - that the commands `convert` et `merge` produce valid 3DTiles.

In order for a MR to be reviewed, the CI must pass completely. Through the following sections, **we will see how to check and correct them beforehand**.

### Check and correct automatically CI issues with pre-commit

Pre-commit is a tool that allows to run a set of checks and corrections before each commit (and push). This tool is not mandatory but highly recommended to simplify the development workflow.

With the pre-commit configuration, the following checks and corrections are made:
 - pyupgrade (corrects directly)
 - autoflake (corrects directly)
 - black (formats directly)
 - flake8 with plugins (alerts only)
 - commitizen (alerts only)

To use it, you must install the development dependencies:
`$ pip install .[dev]`

Then you have to install pre-commit:
`$ pre-commit install -c .pre-commit-config.yaml -f --install-hooks -t pre-push -t pre-commit -t commit-msg`

You can choose not to install the `pre-commit` hooks (by removing `-t pre-commit`) or `pre-commit` itself. But remember that each individual commits of the merge request must pass all these checks. You must rebase correctly your branch for this purpose.

If you want to commit without pre-commit verifications, you need to add the `-n` (or `--no-verify`) flag to the command `git commit`.

Nevertheless, the tests and the verification of type annotations must be executed manually. The execution is rather simple:
`$ pytest` to run all the tests
`$ mypy` to check type annotations

### API documentation

There are no automatic checks yet, so if your modifications change the API, remember to update the examples in the `docs/api.rst` file in order to keep the doc API up to date.

### Commit linter

We use the linter [commitizen](https://github.com/commitizen-tools/commitizen) with the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) configuration (the default one).

### Code linters

We use the linter [flake8](https://flake8.pycqa.org/en/latest/) spiced up with some plugins. In addition, there are [pyupgrade](https://github.com/asottile/pyupgrade) and [autoflake](https://github.com/PyCQA/autoflake).

These linters detect patterns that can be simplified, modernized or that are prone to future bugs. But they also remove useless variables, imports and passes. `Flake8` only raises errors without fixing them. `Pyupgrade` and `autoflake` directly fix found issues.

Some checks are disabled. The whole configuration can be found in the [`.flake8`](./.flake8) file.

### Code formatter

The code of py3dtiles is formatted by [black](https://github.com/ambv/black).

### Type annotations

Typing annotations are verified with [mypy](https://mypy.readthedocs.io/en/stable/). The whole configuration can be found in the `mypy.ini` file. Generic types have been written (to be reused) in the `py3dtiles/typing.py`.

It is strongly discouraged to ignore an error (with the comment `type: ignore`) because it degrades the efficiency of typing. However, if you need to add one, you should specify the ignored error like this:: `# type: ignore [arg-type]` and explain the reason in a comment for future reference.

### Tests

Your changes must be covered by tests as much as possible. We target a 80% coverage. This indicator is monitored through [SonarCloud](https://sonarcloud.io/summary/new_code?id=Oslandia_py3dtiles), and automatically computed by the CI in each merge request.

The CI runs the tests on each supported version of python. Currently, part of the tests are written with the pytest framework and another part with the unittest framework. All new tests must be written with pytest and gradually, the tests written with unittest will be migrated to pytest.

### Checking the validity of generated 3D tiles

This step could be done only with the CI. With the [3d-tiles-validator](https://github.com/CesiumGS/3d-tiles-validator) tool, the job converts 2 point clouds, merges them and checks if the tilesets and tile contents are valid.

### Opening a merge request (MR)

A description must be added explaining the objectives of the MR and the rationale of the changes made. If the MR solves an issue, please specify it in the MR description and in the commit message with a `Fixes #<issue number>`. It's also possible (and very welcomed) to mention related issues (or if the issue is not fully fixed) with `Mention #<issue number>`.
If a CI step failed and you don't know how to fix it, don't hesitate to ask for help!
