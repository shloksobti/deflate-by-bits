#!/usr/bin/env bash
# Download the Ken French daily data used in the paper (49 industries + FF3 factors).
set -euo pipefail
cd "$(dirname "$0")/../data"
base=https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp
for f in 49_Industry_Portfolios_daily_CSV F-F_Research_Data_Factors_daily_CSV; do
  curl -sSL -o "$f.zip" "$base/$f.zip" && unzip -o -q "$f.zip"
done
echo "Note: the paper used the CRSP 2026-08 vintage; later vintages may differ slightly."
