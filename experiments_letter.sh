dataset="letter"
njobs=1
# DP FLBSV

## 2 Clients

clients=2

### 0.5 Fracion
fraction=0.5

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

### 1 Fracion
fraction=1

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

## 5 Clients
clients=5

### 0.5 Fracion
fraction=0.5

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

### 1 Fracion
fraction=1

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

## 10 Clients
clients=10

### 0.5 Fracion
fraction=0.5

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

### 1 Fracion
fraction=1

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

# Ensemble FLBSV

## 2 Clients
clients=2

### 0.5 Fracion
fraction=0.5

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

### 1 Fracion
fraction=1

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

## 5 Clients
clients=5

### 0.5 Fracion
fraction=0.5

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

### 1 Fracion
fraction=1

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

## 10 Clients
clients=10

### 0.5 Fracion
fraction=0.5

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

### 1 Fracion
fraction=1

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync