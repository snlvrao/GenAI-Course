#!/bin/sh
# Run this to open the course with videos working.
# Leave the window open while you read, and close it when you are done.
cd "$(dirname "$0")" || exit 1
exec python3 start.py
