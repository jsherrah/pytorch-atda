"""Main script for ATDA."""

import torch
import torch.nn as nn
from core import domain_adapt, generate_labels, pre_train #, evaluate
from misc import config as cfg
from misc.utils import (enable_cudnn_benchmark, get_data_loader, init_model,
                        init_random_seed)
from models import ClassifierA, EncoderA, EncoderVGG

import aimlTrainPytorch as atr
import argparse
import matplotlib.pyplot as plt

def parseArgs():
    parser = argparse.ArgumentParser(description='Tri-training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataSource', help='path to source dataset')
    parser.add_argument('dataTarget', help='path to target dataset')
    parser.add_argument('--snapshotF',  type=str, default=None, help='optional snapshot to load F from')
    parser.add_argument('--snapshotF1', type=str, default=None, help='optional snapshot to load F1 from')
    parser.add_argument('--snapshotF2', type=str, default=None, help='optional snapshot to load F2 from')
    parser.add_argument('--snapshotFt', type=str, default=None, help='optional snapshot to load Ft from')
    parser.add_argument('--pretrain', action='store_true', help='If true then pre-train the nets')
    parser.add_argument('--domainAdapt', action='store_true', help='If true then perform domain adaptation')

    parser = atr.addParametersDataLoader(parser)

    # for ATR
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')


    return parser.parse_args()


class CompositeModel:
    def __init__(self, ftrs, clfr):
        self.ftrs = ftrs
        self.clfr = clfr

    def eval(self):
        self.ftrs.eval()
        self.clfr.eval()

    def __call__(self, x):
        return self.clfr(self.ftrs(x))

def evaluate(encoder, classifier, data_loader):
    # This works too
    # model = nn.Sequential(encoder, classifier)
    model = CompositeModel(encoder, classifier)
    criterion = nn.CrossEntropyLoss()
    atr.validate(data_loader, model, criterion, args, batchSilent=False)

if __name__ == '__main__':
    print("=== Init ===")
    args = parseArgs()
    # init random seed
    init_random_seed(cfg.manual_seed)
    torch.cuda.set_device(args.gpu)

    model_restore = {
        "F":   args.snapshotF,
        "F_1": args.snapshotF1,
        "F_2": args.snapshotF2,
        "F_t": args.snapshotFt
    }

    # speed up cudnn
    # enable_cudnn_benchmark()

    # load dataset
    if 0:
        source_dataset          = get_data_loader(cfg.source_dataset, get_dataset=True)
        source_data_loader      = get_data_loader(cfg.source_dataset)
        source_data_loader_test = get_data_loader(cfg.source_dataset, train=False)
        target_dataset          = get_data_loader(cfg.target_dataset, get_dataset=True)
        # target_data_loader    = get_data_loader(cfg.target_dataset)
        target_data_loader_test = get_data_loader(cfg.target_dataset, train=False)
        encsize                 = 4*4*48 # 768
    else:
        imgsz, ccropsz = 256, 224
#        imgsz, ccropsz = 28, 28
        source_data_loader, source_data_loader_test, _ = atr.loadDataSets( args,
                                                                           pathAttrName='dataSource',
                                                                           distributed=False,
                                                                           imgSz=imgsz,
                                                                           ccropSz=ccropsz )
        target_data_loader, target_data_loader_test, _ = atr.loadDataSets( args,
                                                                           pathAttrName='dataTarget',
                                                                           distributed=False,
                                                                           imgSz=imgsz,
                                                                           ccropSz=ccropsz )
        source_dataset = source_data_loader.dataset
        target_dataset = target_data_loader.dataset
        encsize = 7*7*512

    if 0:
        plt.figure()
        atr.showDataSet(target_data_loader_test)
        plt.waitforbuttonpress()

    # init models
    #!! F = init_model(net=EncoderA(), restore=model_restore["F"])
    F = init_model(net=EncoderVGG(), restore=model_restore["F"])
    F_1 = init_model(net=ClassifierA(cfg.num_classes, cfg.dropout_keep["F_1"], inputSize=encsize),
                     restore=model_restore["F_1"])
    F_2 = init_model(net=ClassifierA(cfg.num_classes, cfg.dropout_keep["F_2"], inputSize=encsize),
                     restore=model_restore["F_2"])
    F_t = init_model(net=ClassifierA(cfg.num_classes, cfg.dropout_keep["F_t"], inputSize=encsize),
                     restore=model_restore["F_t"])

    # show model structure
    print(">>> F model <<<")
    print(F)
    print(">>> F_1 model <<<")
    print(F_1)
    print(">>> F_2 model <<<")
    print(F_2)
    print(">>> F_t model <<<")
    print(F_t)

    # pre-train on source dataset
    if args.pretrain:
        print("=== Pre-train networks ===")
        pre_train(F, F_1, F_2, F_t, source_data_loader)
        print(">>> evaluate F+F_1")
        evaluate(F, F_1, source_data_loader_test)
        print(">>> evaluate F+F_2")
        evaluate(F, F_2, source_data_loader_test)
        print(">>> evaluate F+F_t")
        evaluate(F, F_t, source_data_loader_test)

    if args.domainAdapt:
        print("=== Adapt F_t ===")
        # generate pseudo labels on target dataset
        print("--- Generate Pseudo Label ---")
        excerpt, pseudo_labels = \
            generate_labels(F, F_1, F_2, target_dataset, cfg.num_target_init, useWeightedSampling=True)
        print(">>> Generate pseudo labels {}".format(
            len(pseudo_labels)))

        # domain adapt between source and target datasets
        print("--- Domain Adapt ---")
        domain_adapt(F, F_1, F_2, F_t,
                     source_dataset, target_dataset, excerpt, pseudo_labels)

    # test F_t on target test dataset
    print("=== Test F_t on target ===")
    evaluate(F, F_t, target_data_loader_test)
    print(">>> evaluate F+F_1 on target")
    evaluate(F, F_1, target_data_loader_test)
    print(">>> evaluate F+F_2 on target")
    evaluate(F, F_2, target_data_loader_test)
#    print(">>> evaluate F+F_t on source")
#    evaluate(F, F_t, source_data_loader_test)
