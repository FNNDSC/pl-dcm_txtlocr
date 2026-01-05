# A Text Locater Plugin

[![Version](https://img.shields.io/docker/v/fnndsc/pl-dcm_txtlocr?sort=semver)](https://hub.docker.com/r/fnndsc/pl-dcm_txtlocr)
[![MIT License](https://img.shields.io/github/license/fnndsc/pl-dcm_txtlocr)](https://github.com/FNNDSC/pl-dcm_txtlocr/blob/main/LICENSE)
[![ci](https://github.com/FNNDSC/pl-dcm_txtlocr/actions/workflows/ci.yml/badge.svg)](https://github.com/FNNDSC/pl-dcm_txtlocr/actions/workflows/ci.yml)

`pl-dcm_txtlocr` is a [_ChRIS_](https://chrisproject.org/) _ds_ plugin which **locates text in DICOM files** and saves it to a specified output directory. It can optionally use GPU acceleration if available.

---

## Abstract

This plugin processes DICOM files, extracts text present in the pixel data of a DICOM file, and creates output files in the desired format. It is designed to work on local machines or inside a ChRIS pipeline and supports GPU acceleration for faster text extraction.

---

## Installation

`pl-dcm_txtlocr` is a _[ChRIS](https://chrisproject.org/) plugin_, meaning it can run either inside ChRIS or from the command line using container technologies like [Apptainer](https://apptainer.org/).

---

## Local Usage

Run locally with Apptainer:

```bash
apptainer exec docker://fnndsc/pl-dcm_txtlocr dcm_txtlocr [--args values...] input/ output/

To print its available options, run:

```shell
apptainer exec docker://fnndsc/pl-dcm_txtlocr dcm_txtlocr --help
```

## Examples

`dcm_txtlocr` requires two positional arguments: a directory containing
input data, and a directory where to create output data.
First, create the input directory and move input data into it.

```shell
mkdir incoming/ outgoing/
mv *.dcm incoming/

apptainer exec docker://fnndsc/pl-dcm_txtlocr:latest dcm_txtlocr incoming/ outgoing/

```
### Using Optional Arguments
```shell
apptainer exec docker://fnndsc/pl-dcm_txtlocr:latest dcm_txtlocr \
    -o text_output \
    -f "*.dcm" \
    -t "txt" \
    -u \
    incoming/ outgoing/
```

## Development

Instructions for developers.

### Building

Build a local container image:

```shell
docker build -t localhost/fnndsc/pl-dcm_txtlocr .
```

### Running

Mount the source code `phi_detector.py` into a container to try out changes without rebuild.

```shell
docker run --rm -it --userns=host -u $(id -u):$(id -g) \
    -v $PWD/phi_detector.py:/usr/local/lib/python3.12/site-packages/dcm_txtlocr.py:ro \
    -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing:rw -w /outgoing \
    localhost/fnndsc/pl-dcm_txtlocr dcm_txtlocr /incoming /outgoing
```

### Testing

Run unit tests using `pytest`.
It's recommended to rebuild the image to ensure that sources are up-to-date.
Use the option `--build-arg extras_require=dev` to install extra dependencies for testing.

```shell
docker build -t localhost/fnndsc/pl-dcm_txtlocr:dev --build-arg extras_require=dev .
docker run --rm -it localhost/fnndsc/pl-dcm_txtlocr:dev pytest
```

## Release

Steps for release can be automated by [Github Actions](.github/workflows/ci.yml).
This section is about how to do those steps manually.

### Increase Version Number

Increase the version number in `setup.py` and commit this file.

### Push Container Image

Build and push an image tagged by the version. For example, for version `1.2.3`:

```
docker build -t docker.io/fnndsc/pl-dcm_txtlocr:1.2.3 .
docker push docker.io/fnndsc/pl-dcm_txtlocr:1.2.3
```

### Get JSON Representation

Run [`chris_plugin_info`](https://github.com/FNNDSC/pl-dcm_txtlocr#usage)
to produce a JSON description of this plugin, which can be uploaded to _ChRIS_.

```shell
docker run --rm docker.io/fnndsc/pl-dcm_txtlocr:1.2.3 chris_plugin_info -d docker.io/fnndsc/pl-dcm_txtlocr:1.2.3 > chris_plugin_info.json
```

Intructions on how to upload the plugin to _ChRIS_ can be found here:
https://chrisproject.org/docs/tutorials/upload_plugin

