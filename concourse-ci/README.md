# Overview

This sets up an example CI pipeline for the invisible cities project. The idea is that, whenever a pull request to a [protected branch](https://help.github.com/en/articles/about-protected-branches) happens:
0. The branch is protected, so the merge cannot be performed until the CI approves.
1. the CI runs the unit tests (currently what your travis CI runs) and errors out if they fail
2. if unit tests pass, the CI submits a job to the majorana cluster that performs sanity checks; a report is generated and put somewhere (eg. a gce bucket)
3. If the sanity checks also pass, the CI marks the PR as approved in github and the merge can now be performed.

I've got 2 pipelines, which is what I generally do:
* the PR pipeline builds pull requests and blocks merges of failed PRs
* another pipeline builds a branch every 24 hours; this way changes in dependencies that break your build get caught early.

I set this up using [concourse](https://concourse-ci.org/) because:
* as a side effect of dragging my current company's devs into the 19th century, I can now set up continuous integration environments based on concourse in my sleep
* concourse is awesome; it does have a learning curve (as do all CI tools) but it's very flexible and forces you to decouple stuff in a way that makes deploying in cloud environments easy.

I left a placeholder script to represent job submission/result gathering from the cluster; this basically means automating via bash whatever you guys do to launch a job and collect the results.

# Usage

## Prerequisites

* [docker-compose](https://www.digitalocean.com/community/tutorials/how-to-install-docker-compose-on-ubuntu-18-04)
* make ie if you're on some unix flavor you're fine

## Walkthrough

These steps must be executed from within this directory:

1. launch the local concourse instance:
  `docker-compose up -d`
2. navigate to [http://localhost:8080](http://localhost:8080) and login with username: test password: test
3. download the [fly cli](https://concourse-ci.org/fly.html) from your local concourse installation
4. log in to concourse via the command line:
  `fly -t local login -c http://localhost:8080`
5. push the build pipeline (the other requires a credential you don't have):
  `make set_local_build_pipeline`

Voilà, you can now unpause the pipeline. The invisible cities tests should run (I'm basically doing the same thing your travis CI is doing).

I'll set up the protected branch and put it up on my gce free account so you guys can access it and test out the workflow.
