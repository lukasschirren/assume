#!/bin/bash
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Writes the study cases of a training campaign into the scenario's config.yaml and lists them in
# assume_gb/hpc/cases.txt, one per line, for train.pbs: one case per training window and seed
# ("learning_<window>_s<seed>"), each with two companions that apply its policies to the whole
# year: "<case>_year" with the policies of the best evaluation episode, "<case>_year_last" with
# those at the end of training. Run it once from the repository root, before submitting and never while jobs
# run (it rewrites config.yaml):
#
#   bash assume_gb/hpc/make_cases.sh                                   # 3 windows x 3 seeds
#   SEEDS="0 1 2 3 4" EPISODES=100 bash assume_gb/hpc/make_cases.sh
#   WINDOWS="sep:2023-09-01:2023-09-29" EXTRA="bidding_strategy_params.max_markup=3" \
#       TAG=m3 bash assume_gb/hpc/make_cases.sh                        # a sweep point on one window
#   SOURCE=learning_cfd APPEND=1 bash assume_gb/hpc/make_cases.sh      # the same, CfD units in the portfolios
#
# A window is label:first delivery day:day after the last delivery day. The simulation starts
# the day before the first delivery, because the market opens 15 hours ahead. EXTRA holds further
# settings for every case (space-separated key=value, as for `python -m assume_gb variant`), TAG
# a suffix for the case names so that campaigns do not overwrite each other. SOURCE is the
# learning case the campaign derives from, with its "<SOURCE>_year": "learning" (default) or
# "learning_cfd", in which the CfD units belong to their companies' portfolios (built with
# --portfolio-cfd); a SOURCE other than "learning" tags the case names with its suffix ("cfd")
# unless TAG is given. APPEND=1 adds to cases.txt instead of replacing it.

set -euo pipefail

PYTHON="${PYTHON:-python}"
YEAR="${YEAR:-2023}"
SEEDS="${SEEDS:-0 1 2}"
EPISODES="${EPISODES:-50}"
WINDOWS="${WINDOWS:-jan:${YEAR}-01-01:${YEAR}-01-15 mar:${YEAR}-03-01:${YEAR}-03-15 sep:${YEAR}-09-01:${YEAR}-09-15}"
EXTRA="${EXTRA:-}"
SOURCE="${SOURCE:-learning}"
TAG="${TAG:-}"
if [ "$SOURCE" != learning ] && [ -z "$TAG" ]; then
    TAG="${SOURCE#learning_}"
fi
CASES_FILE="${CASES_FILE:-assume_gb/hpc/cases.txt}"

mkdir -p assume_gb/results/logs
[ "${APPEND:-0}" = 1 ] || : > "$CASES_FILE"

extra=()
for setting in $EXTRA; do
    extra+=(--set "$setting")
done

for window in $WINDOWS; do
    IFS=: read -r label first end <<< "$window"
    start="$(date -d "$first -1 day" +%F)"
    for seed in $SEEDS; do
        case_name="learning_${label}${TAG:+_$TAG}_s${seed}"
        "$PYTHON" -m assume_gb variant --year "$YEAR" --source "$SOURCE" --case "$case_name" \
            --set "seed=$seed" --set "start_date=$start 00:00" --set "end_date=$end 00:00" \
            --set "markets_config.DA.start_date=$start 09:00" \
            --set "learning_config.training_episodes=$EPISODES" "${extra[@]}"
        "$PYTHON" -m assume_gb variant --year "$YEAR" --source "${SOURCE}_year" --case "${case_name}_year" \
            --set "seed=$seed" "${extra[@]}" \
            --set "learning_config.trained_policies_load_path=learned_strategies/gb_${YEAR}_${case_name}/avg_reward_eval_policies"
        "$PYTHON" -m assume_gb variant --year "$YEAR" --source "${SOURCE}_year" --case "${case_name}_year_last" \
            --set "seed=$seed" "${extra[@]}" \
            --set "learning_config.trained_policies_load_path=learned_strategies/gb_${YEAR}_${case_name}/last_policies"
        echo "$case_name" >> "$CASES_FILE"
    done
done

echo "$(wc -l < "$CASES_FILE") cases in $CASES_FILE: submit with  qsub -J 1-$(wc -l < "$CASES_FILE" | tr -d ' ') -v YEAR=$YEAR,CASES_FILE=$PWD/$CASES_FILE assume_gb/hpc/train.pbs"
