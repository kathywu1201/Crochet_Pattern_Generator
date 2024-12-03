#!/bin/bash
set -e

echo "Container is running!!!"

# Ensure we're in the correct directory
cd /app

if [ "${DEV}" = 1 ]; then
  pipenv shell
else
  pipenv run python cli.py --load # --chunk --embed
fi