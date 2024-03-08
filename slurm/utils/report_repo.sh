#!/bin/bash

echo "========================================================================"
echo "-------- Reporting git repo configuration ------------------------------"
date
echo ""
echo "pwd: $(pwd)"
echo "commit ref: $(git rev-parse HEAD)"
echo ""
git status
echo "========================================================================"
echo ""
