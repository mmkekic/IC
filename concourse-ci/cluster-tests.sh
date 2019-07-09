#!/bin/bash

set -ex

# error out if the credentials haven't been provided
: "${FAKE_CLUSTER_SSH_PRIVATE_KEY:?}"

get_output () {
  : "${ICTDIR:?}"
  : "${OUTDIR:?}"

  cd $ICTDIR
  source manage.sh work_in_python_version_no_tests ${IC_PYTHON_VERSION}

  city irene \
    -i $ICDIR/database/test_data/electrons_40keV_z250_RWF.h5 \
    -o $OUTDIR/irene_result \
    $ICTDIR/invisible_cities/config/irene.conf
    rm -rf /root/miniconda
}

apt-get update && apt install -y curl build-essential

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs

(cd IC && git lfs install && git lfs pull)

mkdir master_out
ICTDIR=`pwd`/IC_master
OUTDIR=`pwd`/master_out
(get_output)
ls -lhtr $OUTDIR

mkdir pr_out
ICTDIR=`pwd`/IC
OUTDIR=pr_out
(get_output)
ls -lhtr $OUTDIR

# echo "Dummy script standing in for the integration tests to be run on the cluster."
# echo "Given credentials passed in via environment variables, this script should post the job to the majorana cluster and gather the run results."
# echo "It should upload the results somewhere accessible by participants (eg. a gcs bucket)."
# echo "It should return a nonzero exit status if the results are erroneous."
# echo "----"
# echo "Be careful with handling the credentials unless you want some script kiddie mining bitcoin on your cluster."
