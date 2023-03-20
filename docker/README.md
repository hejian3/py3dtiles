# Docker

## How build the docker image

You must run the following command in the root folder of the repository
```bash
docker build . -t py3dtiles -f docker/Dockerfile
```

## How to use the docker image

The docker image has a volume on `/app/data/` and the entrypoint is directly the command `py3dtiles`.

#### Examples

Display the help
```bash
docker run -it py3dtiles --rm --help
```

Convert a file into 3d tiles
```bash
mkdir data
cp tests/fixtures/simple.ply data
docker run -it --rm \
    --mount type=bind,source="$(pwd)"/data,target=/app/data/ \
    py3dtiles \
    convert simple.ply
ls data/3dtiles
```
