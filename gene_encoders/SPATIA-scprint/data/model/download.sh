#!/bin/bash

set -ex

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type model --local-dir scPRINT jkobject/scPRINT
