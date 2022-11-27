#!/bin/bash
if [ $# -eq 0 ] || { [ $# -eq 1 ] && [ "$1" = '--python' ]; }; then
  flake8 src/ensembles.py --max-line-length=120 &&
    pylint src/ensembles.py --max-line-length=120 --disable="C0103,C0114,C0115"
fi
if [ $# -eq 0 ] || { [ $# -eq 1 ] && [ "$1" = '--scripts' ]; }; then
  docker run --rm -v "/scripts:/mnt" koalaman/shellcheck:stable build.sh &&
    docker run --rm -v "/scripts:/mnt" koalaman/shellcheck:stable run.sh
fi
if [ $# -eq 0 ] || { [ $# -eq 1 ] && [ "$1" = '--docker' ]; }; then
  cat Dockerfile | docker run --rm -i hadolint/hadolint
fi
