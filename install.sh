#!/bin/bash
poetry build -f wheel -o . && pip uninstall gstwebrtcapp -y && pip install *.whl && rm *.whl