echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd pen-global 1 1 True --njobs 2
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd breast-cancer 1 1 True --njobs 2
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd letter 1 1 True --njobs 2
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd pen-local 1 1 True --njobs 2
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd annthyroid 1 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd satellite 1 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd shuttle 1 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd aloi 1 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd kdd99 1 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"