echo $(date)
start=$(date +%s)
python __main__.py baseline_sklearn pen-global
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_sklearn breast-cancer
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_sklearn letter
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_sklearn pen-local
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_sklearn annthyroid
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_sklearn satellite
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_sklearn shuttle
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_sklearn aloi
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py baseline_sklearn kd99
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"