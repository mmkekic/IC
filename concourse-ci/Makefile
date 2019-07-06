.PHONY: fly_execute
fly_execute:
	fly -t local execute \
		--include-ignored \
		--input IC=../ \
		--config run-tests.yml

.PHONY: set_local_build_pipeline
set_local_build_pipeline:
	fly -t local set-pipeline \
    --pipeline IC-build \
    --config build-pipeline.yml \
		-v FAKE_CLUSTER_SSH_PRIVATE_KEY=fake_key_bro

.PHONY: set_local_pr_pipeline
set_local_pr_pipeline: github_access_token
	fly -t local set-pipeline \
    --pipeline IC-pull-request \
    --config pr-pipeline.yml \
		-v FAKE_CLUSTER_SSH_PRIVATE_KEY=fake_key_bro \
		-v github_access_token=$$(cat github_access_token)
