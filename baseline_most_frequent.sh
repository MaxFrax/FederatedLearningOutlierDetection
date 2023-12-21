echo $(date)
start=$(date +%s)
python __main__.py most_frequent pen-global 1 1 iid
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py most_frequent breast-cancer 1 1 iid
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py most_frequent letter 1 1 iid
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py most_frequent pen-local 1 1 iid
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py most_frequent annthyroid 1 1 iid
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py most_frequent satellite 1 1 iid
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py most_frequent shuttle 1 1 iid
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py most_frequent aloi 1 1 iid
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py most_frequent kdd99 1 1 iid
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync