# 5 Clients 1.0 Fraction
echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-global 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv breast-cancer 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv letter 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-local 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv annthyroid 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv satellite 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv shuttle 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv aloi 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv kdd99 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

# 5 Clients .5 Fraction
echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-global 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv breast-cancer 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv letter 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-local 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv annthyroid 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv satellite 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv shuttle 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv aloi 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv kdd99 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"