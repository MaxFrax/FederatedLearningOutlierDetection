import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import logging
import sys
from sklearn.svm import OneClassSVM
from scipy.stats import uniform
from flod.classifiers.bsvclassifier import BSVClassifier
from flod.classifiers.federatedbsvclassifier import FederatedBSVClassifier
from experiments.experiments import compute_baseline, get_datasets, compute_federated_experiment

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

# create the parser object
parser = argparse.ArgumentParser(
    description='Run the experiments from the paper.')

# add arguments to the parser
# TODO add support for federated learning biased experiments
parser.add_argument('experiment', type=str, choices=[
                    'baseline_sklearn', 'baseline_svdd', 'flbsv_precomputed_gamma', 'flbsv_precomputed_cos'], help='The experiment to run.')

parser.add_argument('--njobs', type=int,
                    help='Number of parallel threads to use.', default=-1)

parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING',
                    'ERROR', 'CRITICAL'], help='The logging level.', dest='loglevel', default='INFO')

parser.add_argument('dataset', type=str, choices=get_datasets().keys(), help='The dataset to run the experiment on.')

# parse the arguments
args = parser.parse_args()

def print_results(results):
    auc_df, avg_df, acc_df = results
    print('AUC')
    print(auc_df)
    print('Average Precision')
    print(avg_df)
    print('Accuracy')
    print(acc_df)

def baseline_sklearn():
    logger.info('Running baseline sklearn experiment on %s', args.dataset)

    classifier = OneClassSVM(kernel='rbf')
    distributions = dict(nu=uniform(loc=0.2, scale=0.8),
                         gamma=uniform(loc=0, scale=3))
    
    print_results(compute_baseline('sklearn_{}.csv', classifier, distributions, args.dataset, args.njobs))

def baseline_svdd():
    logger.info('Running baseline svdd experiment on %s', args.dataset)

    classifier = BSVClassifier(normal_class_label=1, outlier_class_label=-1)
    distributions = {'c':uniform(loc=0.2, scale=0.8),'q':uniform(loc=0, scale=3)}
    
    print_results(compute_baseline('svdd_{}.csv', classifier, distributions, args.dataset, args.njobs))

def flbsv_precomputed_gamma():
    logger.info('Running flbsv experiment with precomputed gamma on %s', args.dataset)

    classifier = FederatedBSVClassifier(method='gamma', normal_class_label=1, outlier_class_label=-1, max_rounds=1)
    distributions = {'C':uniform(loc=0.2, scale=0.8),'q':uniform(loc=0, scale=3)}
    
    print_results(compute_federated_experiment('flbsv_precomputed_gamma_{}.csv', classifier, distributions, args.dataset, args.njobs))

    

def flbsv_precomputed_cos():
    logger.info('Running flbsv experiment with precomputed cos on %s', args.dataset)

    classifier = FederatedBSVClassifier(method='cos', normal_class_label=1, outlier_class_label=-1, max_rounds=1)
    distributions = {'C':uniform(loc=0.2, scale=0.8),'q':uniform(loc=0, scale=3)}
    
    print_results(compute_federated_experiment('flbsv_precomputed_cos_{}.csv', classifier, distributions, args.dataset, args.njobs))

if args.loglevel:
    logger.setLevel(args.loglevel)

if args.experiment == 'baseline_sklearn':
    baseline_sklearn()
elif args.experiment == 'baseline_svdd':
    baseline_svdd()
elif args.experiment == 'flbsv_precomputed_gamma':
    flbsv_precomputed_gamma()
elif args.experiment == 'flbsv_precomputed_cos':
    flbsv_precomputed_cos()