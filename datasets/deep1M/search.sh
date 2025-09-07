#!/bin/bash
l_values=(100 150 175 200 250 300 400 500 600 700 800 900)
total_query=1000
per_query_num=1000

for l in "${l_values[@]}"; do
  for ((start_query = 0; start_query < total_query; start_query += per_query_num)); do

    ../../cmake-build/BAMG "search" "BAMG" "deep1M" 50 60 500 "BNP" "$l" "$per_query_num" "$start_query"

  done
  echo "-----------------------------"
done