import warnings

from flod.classifiers.dp_flbsv import DPFLBSV
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import logging
import sys
from sklearn.dummy import DummyClassifier
from sklearn.svm import OneClassSVM
from scipy.stats import uniform
from flod.classifiers.bsvclassifier import BSVClassifier
from flod.classifiers.federatedbsvclassifier import FederatedBSVClassifier
from experiments.experiments import compute_baseline, get_datasets, compute_federated_experiment
from flod.classifiers.ensemble_flbsv import EnsembleFLBSV
import os

import neptune
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

run = neptune.init_run(
    project="maxfrax/FlodThesis",
    api_token=os.environ.get("NEPTUNE_API_TOKEN"),
)

# create the parser object
parser = argparse.ArgumentParser(
    description='Run the experiments from the paper.')

# add arguments to the parser

parser.add_argument('experiment', type=str, choices=[
                    'baseline_sklearn', 'baseline_svdd', 'dp_flbsv', 'dp_flbsv_noisy', 'ensemble_flbsv', 'ensemble_flbsv_noisy', 'most_frequent'], help='The experiment to run.')

parser.add_argument('--njobs', type=int,
                    help='Number of parallel threads to use.', default=-1)

parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING',
                    'ERROR', 'CRITICAL'], help='The logging level.', dest='loglevel', default='INFO')

parser.add_argument('dataset', type=str, choices=get_datasets().keys(), help='The dataset to run the experiment on.')

parser.add_argument('clients_amount', type=int, help='The amount of clients to use in the federated learning experiment.')

parser.add_argument('client_fraction', type=float, help='The fraction of clients to use in the federated learning experiment.')

parser.add_argument('iid_dataset', type=str, choices=['iid', 'biased'], help='Whether to use an iid dataset or not.')

parser.add_argument('evaluation_technique', type=str, choices=['unsupervised', 'nested_crossval'], help='The evaluation technique to use.')

def print_results(results):
    pretty_auc = f'{results["roc_auc"]["mean"]:.4f} ± {results["roc_auc"]["std"]:.4f}'
    pretty_acc = f'{results["accuracy"]["mean"]:.4f} ± {results["accuracy"]["std"]:.4f}'
    print(f'ROC AUC: {pretty_auc}')
    print(f'Accuracy: {pretty_acc}')

    run["results/avg_auc"]=results["roc_auc"]["mean"]
    run["results/avg_acc"]=results["accuracy"]["mean"]
    run["results/std_auc"]=results["roc_auc"]["std"]
    run["results/std_acc"]=results["accuracy"]["std"]
    run["results/pretty_auc"]=pretty_auc
    run["results/pretty_acc"]=pretty_acc

def baseline_sklearn():
    logger.info('Running baseline sklearn experiment on %s', args.dataset)

    if args.clients_amount != 1:
        raise ValueError('Baseline sklearn experiment does not support clients amount different than 1.')
    
    if args.iid_dataset == 'biased':
        raise ValueError('Baseline sklearn experiment does not support non iid datasets.')

    classifier = OneClassSVM(kernel='rbf', gamma=1.0)
    distributions = dict(nu=uniform(loc=0.2, scale=0.8))
    print_results(compute_baseline(classifier, distributions, args.dataset, args.njobs, args.iid_dataset, args.evaluation_technique))

def baseline_svdd():
    logger.info('Running baseline svdd experiment on %s', args.dataset)

    if args.clients_amount != 1:
        raise ValueError('Baseline sklearn experiment does not support clients amount different than 1.')
    
    if args.iid_dataset == 'biased':
        raise ValueError('Baseline sklearn experiment does not support non iid datasets.')

    classifier = BSVClassifier(normal_class_label=1, outlier_class_label=-1, q=1)
    distributions = {'c':uniform(loc=0.2, scale=0.8)}
    print_results(compute_baseline(classifier, distributions, args.dataset, args.njobs, args.iid_dataset, args.evaluation_technique))

def dp_flbsv():
    logger.info('Running dp flbsv experiment on %s', args.dataset)

    classifier = DPFLBSV(-1, -1, normal_class_label=1, outlier_class_label=-1, max_rounds=1, q=1, total_clients=args.clients_amount, client_fraction=args.client_fraction)
    distributions = {'C':uniform(loc=0.2, scale=0.8)}
    
    print_results(compute_federated_experiment(classifier, distributions, args.dataset, args.njobs, args.iid_dataset, args.evaluation_technique))

def dp_flbsv_noisy():
    logger.info('Running dp flbsv noisy experiment on %s', args.dataset)

    classifier = DPFLBSV(1, .001, normal_class_label=1, outlier_class_label=-1, max_rounds=1, q=1, total_clients=args.clients_amount, client_fraction=args.client_fraction)
    distributions = {'C':uniform(loc=0.2, scale=0.8)}
    
    print_results(compute_federated_experiment(classifier, distributions, args.dataset, args.njobs, args.iid_dataset, args.evaluation_technique))

def ensemble_flbsv():
    logger.info('Running ensemble flbsv experiment on %s', args.dataset)

    classifier = EnsembleFLBSV(normal_class_label=1, outlier_class_label=-1, q=1, total_clients=args.clients_amount, client_fraction=args.client_fraction)
    distributions = {'C':uniform(loc=0.2, scale=0.8)}
    
    print_results(compute_federated_experiment(classifier, distributions, args.dataset, args.njobs, args.iid_dataset, args.evaluation_technique))

def ensemble_flbsv_noisy():
    logger.info('Running ensemble flbsv noisy experiment on %s', args.dataset)

    classifier = EnsembleFLBSV(normal_class_label=1, outlier_class_label=-1, privacy=True, q=1, total_clients=args.clients_amount, client_fraction=args.client_fraction)
    distributions = {'C':uniform(loc=0.2, scale=0.8)}
    
    print_results(compute_federated_experiment(classifier, distributions, args.dataset, args.njobs, args.iid_dataset, args.evaluation_technique))

def most_frequent():
    logger.info('Running dummy most frequent experiment on %s', args.dataset)

    classifier = DummyClassifier(strategy='most_frequent')
    distributions = {}

    print_results(compute_baseline(classifier, distributions, args.dataset, args.njobs, args.iid_dataset, args.evaluation_technique))

# parse the arguments
args = parser.parse_args()
run["parameters"] = vars(args)
run["sys/tags"].add([args.experiment, args.dataset, args.iid_dataset, args.evaluation_technique, "unsupervised"])

if args.loglevel:
    logger.setLevel(args.loglevel)

if args.client_fraction <= 0 or args.client_fraction > 1:
    raise ValueError('Client fraction must be between 0 and 1')

if args.experiment == 'baseline_sklearn':
    baseline_sklearn()
elif args.experiment == 'baseline_svdd':
    baseline_svdd()
elif args.experiment == 'dp_flbsv':
    dp_flbsv()
elif args.experiment == 'dp_flbsv_noisy':
    dp_flbsv_noisy()
elif args.experiment == 'ensemble_flbsv':
    ensemble_flbsv()
elif args.experiment == 'ensemble_flbsv_noisy':
    ensemble_flbsv_noisy()
elif args.experiment == 'most_frequent':
    most_frequent()

if args.evaluation_technique == 'unsupervised':
    run["results/crossvalidation"].upload('cv_results.csv')
run.stop()