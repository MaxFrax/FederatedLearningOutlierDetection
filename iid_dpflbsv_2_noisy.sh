echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy pen-global 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy pen-global 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy breast-cancer 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy breast-cancer 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy letter 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy letter 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy pen-local 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy pen-local 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy annthyroid 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy annthyroid 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy satellite 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy satellite 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy shuttle 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy shuttle 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy aloi 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy aloi 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy kdd99 2 1 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy kdd99 2 .5 True --njobs 1
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync