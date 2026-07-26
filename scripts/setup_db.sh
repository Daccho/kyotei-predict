#!/usr/bin/env bash
# Bring up the local database and apply the schema.
#
# The web container is ephemeral: Postgres is re-created on every session, so
# this script must be safe to re-run. Schema and seed are both idempotent.
set -euo pipefail

DB_NAME="${KYOTEI_DB_NAME:-kyotei}"
DB_USER="${KYOTEI_DB_USER:-kyotei}"
DB_PASS="${KYOTEI_DB_PASSWORD:-kyotei}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! pg_isready -q; then
    echo "starting postgres cluster..."
    pg_ctlcluster 16 main start
fi

run_pg() { su postgres -c "psql -v ON_ERROR_STOP=1 $*"; }

echo "==> ensuring role and database"
su postgres -c "psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'\"" | grep -q 1 \
    || su postgres -c "psql -q -c \"CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}'\""
su postgres -c "psql -tAc \"SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'\"" | grep -q 1 \
    || su postgres -c "createdb -O ${DB_USER} ${DB_NAME}"

echo "==> applying schema"
su postgres -c "psql -v ON_ERROR_STOP=1 -q -d ${DB_NAME} -f ${REPO_ROOT}/sql/schema.sql"
su postgres -c "psql -v ON_ERROR_STOP=1 -q -d ${DB_NAME} -f ${REPO_ROOT}/sql/seed_stadiums.sql"
su postgres -c "psql -q -d ${DB_NAME} -c \
    'GRANT ALL ON ALL TABLES IN SCHEMA public TO ${DB_USER}; \
     GRANT ALL ON SCHEMA public TO ${DB_USER};'"

echo "==> done. tables:"
su postgres -c "psql -d ${DB_NAME} -c '\\dt'"
echo "==> views:"
su postgres -c "psql -d ${DB_NAME} -c '\\dv'"
echo
echo "DSN: postgresql://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}"
