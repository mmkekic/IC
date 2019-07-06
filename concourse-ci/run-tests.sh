#!/bin/bash

set -e

apt-get update && apt install -y curl build-essential

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
git lfs install

source manage.sh work_in_python_version_no_tests ${IC_PYTHON_VERSION}

HYPOTHESIS_PROFILE=hard bash manage.sh run_tests_par 2
