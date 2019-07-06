#!/bin/bash

set -e

# error out if the credentials haven't been provided
: "${FAKE_CLUSTER_SSH_PRIVATE_KEY:?}"

echo "Dummy script standing in for the integration tests to be run on the cluster."
echo "Given credentials in passed in via environment variables, this script should post the job to the majorana cluster and gather the run results."
echo "It should upload the results somewhere accessible by participants (eg. a gcs bucket)."
echo "It should return a nonzero exit status if the results are erroneous."
echo "----"
echo "Be careful with handling the credentials unless you want some script kiddie mining bitcoin on your cluster."
