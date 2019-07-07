.PHONY: fly_remote_login
fly_remote_login:
	fly -t remote login \
	  -c https://ci.ific-invisible-cities.com

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
set_local_pr_pipeline: credentials/github_access_token
	fly -t local set-pipeline \
    --pipeline IC-pull-request \
    --config pr-pipeline.yml \
		-v FAKE_CLUSTER_SSH_PRIVATE_KEY=fake_key_bro \
		-v github_access_token=$$(cat credentials/github_access_token)

.PHONY: set_build_pipeline
set_build_pipeline:
	fly -t remote set-pipeline \
    --pipeline IC-build \
    --config build-pipeline.yml \
		-v FAKE_CLUSTER_SSH_PRIVATE_KEY=fake_key_bro

.PHONY: set_pr_pipeline
set_pr_pipeline: credentials/github_access_token
	fly -t remote set-pipeline \
    --pipeline IC-pull-request \
    --config pr-pipeline.yml \
		-v FAKE_CLUSTER_SSH_PRIVATE_KEY=fake_key_bro \
		-v github_access_token=$$(cat credentials/github_access_token)


.PHONY: launch_prod_concourse
launch_prod_concourse:
	docker-compose -f prod-docker-compose.yml down -v --remove-orphans
	docker-compose -f prod-docker-compose.yml run make_dummy_certs
	docker-compose -f prod-docker-compose.yml up -d nginx
	docker-compose -f prod-docker-compose.yml run erase_certs
	docker-compose -f prod-docker-compose.yml run create_certs
	docker-compose -f prod-docker-compose.yml exec nginx nginx -s reload
