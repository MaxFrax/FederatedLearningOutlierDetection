# 2 Clients 1.0 Fraction
echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-global 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv breast-cancer 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv letter 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-local 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv annthyroid 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv satellite 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv shuttle 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv aloi 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv kdd99 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

# 2 Clients .5 Fraction
echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-global 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv breast-cancer 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv letter 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-local 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv annthyroid 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv satellite 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv shuttle 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv aloi 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv kdd99 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

# 2 Clients .2 Fraction
echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-global 2 .2 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv breast-cancer 2 .2 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv letter 2 .2 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv pen-local 2 .2 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv annthyroid 2 .2 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv satellite 2 .2 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv shuttle 2 .2 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv aloi 2 .2 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv kdd99 2 .2 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"