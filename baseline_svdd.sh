echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd pen-global --njobs 6
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd breast-cancer --njobs 6
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd letter --njobs 6
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd pen-local --njobs 6
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd annthyroid --njobs 6
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd satellite --njobs 6
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd shuttle --njobs 6
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd aloi --njobs 6
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd kd99 --njobs 6
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"