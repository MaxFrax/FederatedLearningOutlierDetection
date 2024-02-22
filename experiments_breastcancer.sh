dataset="breast-cancer"
njobs=-1
tech="unsupervised"

# Baselines

echo $(date)
start=$(date +%s)
python __main__.py baseline_sklearn $dataset 1 1 iid --njobs $njobs $tech
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

echo $(date)
start=$(date +%s)
python __main__.py baseline_svdd $dataset 1 1 iid --njobs $njobs $tech
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

# DP FLBSV

## 2 Clients

clients=2

### 0.5 Fracion
fraction=0.5

#### iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
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
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
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
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
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
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
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
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
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
python __main__.py dp_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py dp_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
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
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
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
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
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
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
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
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
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
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
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
python __main__.py ensemble_flbsv $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync


#### iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction iid --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid no noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync

#### non-iid noise

echo $(date)
start=$(date +%s)
python __main__.py ensemble_flbsv_noisy $dataset $clients $fraction biased --njobs $njobs $tech
end=$(date +%s)
end=$(date +%s)
echo "Elapsed time: $(($end-$start)) s"

export $(cat .env | xargs)
neptune sync