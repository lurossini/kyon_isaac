#!/bin/bash

set -e

# cd to script dir
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

docker compose up -d
docker compose exec kyon-isaac-base bash