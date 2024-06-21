#!/bin/bash
poetry build -f wheel -o . && pip install --force-reinstall *.whl && rm *.whl