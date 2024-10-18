#!/usr/bin/env bash
if hash podman 2>/dev/null; then
  CMD=podman
elif hash nerdctl 2>/dev/null; then
  CMD=nerdctl
elif hash docker 2>/dev/null; then
  CMD=docker
else
  echo "No container runtime found. Please install podman, nerdctl or docker."
  exit 1
fi

DATA_DIR=$HOME/neo4j/data
LOG_DIR=$HOME/neo4j/logs
PLUGIN_DIR=$HOME/neo4j/plugins
BACKUP_DIR=$HOME/neo4j/backups
# Assure thetthe directories exist
if [ ! -d "$DATA_DIR" ]; then
  mkdir -p $DATA_DIR
fi
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p $LOG_DIR
fi
if [ ! -d "$PLUGIN_DIR" ]; then
  mkdir -p $PLUGIN_DIR
fi
if [ ! -d "$BACKUP_DIR" ]; then
  mkdir -p $BACKUP_DIR
fi

# Start container
$CMD run \
  --publish=7474:7474 --publish=7687:7687 \
  --volume=$HOME/neo4j/data:/data \
  --volume=$HOME/neo4j/logs:/logs \
  --volume=$HOME/neo4j/plugins:/plugins \
  --volume=$HOME/neo4j/backups:/backups \
  --user="$(id -u):$(id -g)" \
  --name=neo4j \
  --env NEO4J_AUTH=none \
  --env NEO4J_apoc_export_file_enabled=true \
  --env NEO4J_apoc_import_file_enabled=true \
  --env NEO4J_apoc_import_file_use__neo4j__config=true \
  --env NEO4JLABS_PLUGINS='["apoc"]' \
  --detach \
  neo4j:latest

