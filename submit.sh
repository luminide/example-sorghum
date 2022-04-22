#!/bin/bash -e
#
# Make a kaggle submission

echo
echo $(tput -T xterm setaf 4)Press enter to upload submission.csv to Kaggle$(tput -T xterm sgr0)
read

# check for Kaggle API credentials
[[ ! -f ~/.kaggle/kaggle.json ]]  && { echo 'error: Kaggle API token needs to be configured using the "Import Data" tab'; exit 1; }

set -x
kaggle competitions submit -c sorghum-id-fgvc-9 -f submission.csv -m "$(date)"
set +x
echo
