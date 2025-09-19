#!/usr/bin/env bash

# create root-ca
certtool \
	--generate-privkey \
	--outfile ca-priv.pem

TEMPLATE="cn=root-ca\nca\ncert_signing_key\nexpiration_days = 3650\n"

certtool \
	--generate-self-signed \
	--template <(echo -e "${TEMPLATE}") \
	--outfile ca-cert.pem \
	--load-privkey ca-priv.pem

# create intermediate-ca
certtool \
	--generate-privkey \
	--outfile int-ca-priv.pem

TEMPLATE="cn=intermediate-ca\nca\ncert_signing_key\nexpiration_days = 3650\n"

certtool \
	--generate-certificate \
	--template <(echo -e "${TEMPLATE}") \
	--outfile int-ca-cert.pem \
	--load-privkey int-ca-priv.pem \
	--load-ca-privkey ca-priv.pem \
	--load-ca-certificate ca-cert.pem

# code-signing key
certtool \
	--generate-privkey \
	--key-type ecdsa \
	--curve secp384r1 \
	--outfile signing-key.pem

TEMPLATE="cn=intermediate-ca\nca\ncode_signing_key\nsigning_key\nexpiration_days = 3650\n"

certtool \
	--generate-certificate \
	--template <(echo -e "${TEMPLATE}") \
	--outfile signing-key-cert.pem \
	--load-privkey signing-key.pem \
	--load-ca-privkey int-ca-priv.pem \
	--load-ca-certificate int-ca-cert.pem

certtool \
	--pubkey-info \
	--load-privkey signing-key.pem \
	> signing-key-pub.pem