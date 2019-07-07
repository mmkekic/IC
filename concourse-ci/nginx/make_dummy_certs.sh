#!/bin/sh

set -e

CERTS=/etc/letsencrypt/live/ci.ific-invisible-cities.com

mkdir -p $CERTS

openssl req -x509 -nodes -newkey rsa:1024 -days 1 \
  -keyout $CERTS/privkey.pem \
  -out $CERTS/fullchain.pem \
  -subj '/CN=localhost'
