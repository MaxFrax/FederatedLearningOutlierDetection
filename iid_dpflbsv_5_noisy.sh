echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy pen-global 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy pen-global 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy breast-cancer 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy breast-cancer 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy letter 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy letter 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy pen-local 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy pen-local 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy annthyroid 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy annthyroid 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy satellite 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy satellite 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy shuttle 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy shuttle 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy aloi 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy aloi 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy kdd99 5 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy kdd99 5 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync