echo $(date)
start=$(date +%s)
python __main__.py flbsv_precomputed_gamma pen-global --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py flbsv_precomputed_gamma breast-cancer --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py flbsv_precomputed_gamma letter --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py flbsv_precomputed_gamma pen-local --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py flbsv_precomputed_gamma annthyroid --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo "WARNING: this is not a full evaluation. Some datasets missing"