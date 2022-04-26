#!/bin/bash
# Bash Script for Docker Container
coverage run -m unittest unit_tests.py
coverage report
echo Tests Complete!