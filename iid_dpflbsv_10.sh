# 10 Clients 1.0 Fraction
echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-global 10 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv breast-cancer 10 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv letter 10 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-local 10 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv annthyroid 10 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv satellite 10 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv shuttle 10 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv aloi 10 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv kdd99 10 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

# 10 Clients .5 Fraction
echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-global 10 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv breast-cancer 10 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv letter 10 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-local 10 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv annthyroid 10 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv satellite 10 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv shuttle 10 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv aloi 10 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv kdd99 10 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"