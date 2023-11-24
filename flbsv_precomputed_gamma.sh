echo $(date)
start=$(date +%s)
python __main__.py flbsv_precomputed_gamma pen-global --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"