"""Train script for ATDA."""

import sys
import torch
from torch import nn
from torch.utils.data import ConcatDataset, TensorDataset

from datasets import get_dummy
from misc import config as cfg
from misc.utils import (calc_similiar_penalty, get_inf_iterator, get_optimizer,
                        get_sampled_data_loader, guess_pseudo_labels,
                        make_data_loader, make_variable, save_model)
import matplotlib.pyplot as plt
import aimlTrainPytorch as atr
import numpy as np

def pre_train(F, F_1, F_2, F_t, source_data):
    """Pre-train models on source domain dataset."""
    # set train state for Dropout and BN layers
    F.train()
    F_1.train()
    F_2.train()
    F_t.train()

    # set criterion for classifier and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_F = get_optimizer(F, "Adam")
    optimizer_F_1 = get_optimizer(F_1, "Adam")
    optimizer_F_2 = get_optimizer(F_2, "Adam")
    optimizer_F_t = get_optimizer(F_t, "Adam")

    # start training
    for epoch in range(cfg.num_epochs_pre):
        for step, (images, labels) in enumerate(source_data):
            # convert into torch.autograd.Variable
            images = make_variable(images)
            labels = make_variable(labels)

            # zero-grad optimizer
            optimizer_F.zero_grad()
            optimizer_F_1.zero_grad()
            optimizer_F_2.zero_grad()
            optimizer_F_t.zero_grad()

            # forward networks
            out_F = F(images)
            out_F_1 = F_1(out_F)
            out_F_2 = F_2(out_F)
            out_F_t = F_t(out_F)

            # compute loss
            loss_similiar = calc_similiar_penalty(F_1, F_2)
            loss_F_1 = criterion(out_F_1, labels)
            loss_F_2 = criterion(out_F_2, labels)
            loss_F_t = criterion(out_F_t, labels)
            loss_F = loss_F_1 + loss_F_2 + loss_F_t + loss_similiar
            loss_F.backward()

            # optimize
            optimizer_F.step()
            optimizer_F_1.step()
            optimizer_F_2.step()
            optimizer_F_t.step()

            # print step info
            if ((step + 1) % cfg.log_step == 0):
                print("Epoch [{}/{}] Step[{}/{}] Loss("
                      "Total={:.5f} F_1={:.5f} F_2={:.5f} "
                      "F_t={:.5f} sim={:.5f})"
                      .format(epoch + 1,
                              cfg.num_epochs_pre,
                              step + 1,
                              len(source_data),
                              loss_F.item(), #.data[0],
                              loss_F_1.item(), #.data[0],
                              loss_F_2.item(), #.data[0],
                              loss_F_t.item(), #.data[0],
                              loss_similiar.item(), #.data[0],
                              ))

        # save model
        if ((epoch + 1) % cfg.save_step == 0):
            save_model(F, "pretrain-F-{}.pt".format(epoch + 1))
            save_model(F_1, "pretrain-F_1-{}.pt".format(epoch + 1))
            save_model(F_2, "pretrain-F_2-{}.pt".format(epoch + 1))
            save_model(F_t, "pretrain-F_t-{}.pt".format(epoch + 1))

    # save final model
    save_model(F, "pretrain-F-final.pt")
    save_model(F_1, "pretrain-F_1-final.pt")
    save_model(F_2, "pretrain-F_2-final.pt")
    save_model(F_t, "pretrain-F_t-final.pt")


# Returns excerpt, index into target_dataset.
def generate_labels(F, F_1, F_2, target_dataset, num_target, useWeightedSampling=False):
    """Genrate pseudo labels for target domain dataset."""
    # set eval state for Dropout and BN layers
    F.eval()
    F_1.eval()
    F_2.eval()

    # get candidate samples
    print("Num of sampled target data: {}".format(num_target))

    # get sampled data loader
    if useWeightedSampling:
        classCounts, target_labels = atr.getClassCountsOfDataSet(target_dataset)
        assert np.all(classCounts > 0)
        classWeights = 1.0 / classCounts.astype(float)
        sampleWeights = classWeights[target_labels]
        target_sampler = torch.utils.data.sampler.WeightedRandomSampler( sampleWeights,
                                                                         num_target,
                                                                        replacement=True )
        data_loader = make_data_loader(target_dataset, sampler=target_sampler, shuffle=False)
    else:
        data_loader = get_sampled_data_loader(target_dataset,
                                              num_target, shuffle=True)

    # get output of F_1 and F_2 on sampled target dataset
    out_F_1_total = None
    out_F_2_total = None
    for step, (images, _) in enumerate(data_loader):
        # convert into torch.autograd.Variable
        images = make_variable(images)
        # forward networks
        out_F = F(images)
        out_F_1 = F_1(out_F)
        out_F_2 = F_2(out_F)

        # concat outputs
        if step == 0:
            out_F_1_total = out_F_1.data.cpu()
            out_F_2_total = out_F_2.data.cpu()
        else:
            out_F_1_total = torch.cat(
                [out_F_1_total, out_F_1.data.cpu()], 0)
            out_F_2_total = torch.cat(
                [out_F_2_total, out_F_2.data.cpu()], 0)

    # guess pseudo labels
    excerpt, pseudo_labels = \
        guess_pseudo_labels(out_F_1_total, out_F_2_total)

    assert len(excerpt) <= num_target
    assert np.all(np.logical_and(excerpt>= 0, excerpt < len(target_dataset)))
    print('Max excerpt = {}, but length of target data set = {}'.format(excerpt.max(), len(target_dataset)))

    return excerpt, pseudo_labels


def domain_adapt(F, F_1, F_2, F_t,
                 source_dataset, target_dataset,
                 excerpt, pseudo_labels):
    """Perform Doamin Adaptation between source and target domains."""
    # set criterion for classifier and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_F = get_optimizer(F, "Adam")
    optimizer_F_1 = get_optimizer(F_1, "Adam")
    optimizer_F_2 = get_optimizer(F_2, "Adam")
    optimizer_F_t = get_optimizer(F_t, "Adam")

    # get labelled target dataset
    print('pseudo_labels = %s' % str(pseudo_labels))
    target_dataset_labelled = get_dummy(
        target_dataset, excerpt, pseudo_labels, get_dataset=True)

    # merge soruce data and target data
    merged_dataset = ConcatDataset([source_dataset,
                                    target_dataset_labelled])

    print('target_dataset_labelled = %d' % len(target_dataset_labelled))

    # start training
    for k in range(cfg.num_epochs_k):
        # set train state for Dropout and BN layers
        F.train()
        F_1.train()
        F_2.train()
        F_t.train()

        merged_dataloader = make_data_loader(merged_dataset)
        target_dataloader_labelled = make_data_loader(target_dataset_labelled)
        target_dataloader_labelled_iter = get_inf_iterator(target_dataloader_labelled)

        if 0:
            plt.figure()
            atr.showDataSet(target_dataloader_labelled)
            plt.waitforbuttonpress()

        if 0:
            # There's a bug here, the labels are not the same data type.  print them out!!
            source_dataloader_iter = get_inf_iterator(make_data_loader(source_dataset))

            a, b = next(source_dataloader_iter)
            c, d = next(target_dataloader_labelled_iter)
            print('source labels = {}'.format(b))
            print('target labels = {}'.format(d))
            sys.exit(0)

        for epoch in range(cfg.num_epochs_adapt):
            for step, rez in enumerate(merged_dataloader):
                #!!print('rez = %s' % rez)
                images, labels = rez
                if images.shape[0] < cfg.batch_size:
                    print('WARNING: batch of size %d smaller than desired %d: skipping' % \
                          (images.shape[0], cfg.batch_size))
                    continue

                # sample from T_l
                images_tgt, labels_tgt = next(target_dataloader_labelled_iter)
                while images_tgt.shape[0] < cfg.batch_size:
                    print('WARNING: target batch of size %d smaller than desired %d' % \
                          (images_tgt.shape[0], cfg.batch_size))
                    images_tgt, labels_tgt = next(target_dataloader_labelled_iter)

                # convert into torch.autograd.Variable
                images = make_variable(images)
                labels = make_variable(labels)
                images_tgt = make_variable(images_tgt)
                labels_tgt = make_variable(labels_tgt)

                # zero-grad optimizer
                optimizer_F.zero_grad()
                optimizer_F_1.zero_grad()
                optimizer_F_2.zero_grad()
                optimizer_F_t.zero_grad()

                # forward networks
                #print('images shape = {}'.format(images.shape))#!!
                out_F = F(images)
                #print('out_F = {}'.format(out_F.shape))#!!
                out_F_1 = F_1(out_F)
                out_F_2 = F_2(out_F)
                out_F_t = F_t(F(images_tgt))

                # compute labelling loss
                loss_similiar = calc_similiar_penalty(F_1, F_2)
                loss_F_1 = criterion(out_F_1, labels)
                loss_F_2 = criterion(out_F_2, labels)
                loss_labelling = loss_F_1 + loss_F_2 + loss_similiar
                loss_labelling.backward()

                # compute target specific loss
                loss_F_t = criterion(out_F_t, labels_tgt)
                loss_F_t.backward()

                # optimize
                optimizer_F.step()
                optimizer_F_1.step()
                optimizer_F_2.step()
                optimizer_F_t.step()

                # print step info
                if ((step + 1) % cfg.log_step == 0):
                    print("K[{}/{}] Epoch [{}/{}] Step[{}/{}] Loss("
                          "labelling={:.5f} target={:.5f})"
                          .format(k + 1,
                                  cfg.num_epochs_k,
                                  epoch + 1,
                                  cfg.num_epochs_adapt,
                                  step + 1,
                                  len(merged_dataloader),
                                  loss_labelling.item(), #.data[0],
                                  loss_F_t.item(), #.data[0],
                                  ))
                #!!print('end of loop')

        # re-compute the number of selected taget data
        num_target = (k + 2) * len(source_dataset) // 20
        num_target = min(num_target, cfg.num_target_max)
        print(">>> Set num of sampled target data: {}".format(num_target))

        # re-generate pseudo labels
        excerpt, pseudo_labels = generate_labels(
            F, F_1, F_2, target_dataset, num_target, useWeightedSampling=True)
        print(">>> Genrate pseudo labels [{}] numtarget = {}".format(
            len(target_dataset_labelled), num_target))

        print('sizes = {}, {}, excerpt = {}, \npseudo_labels = {}'.format(len(excerpt), len(pseudo_labels),
                                                                          excerpt, pseudo_labels))

        # get labelled target dataset
        target_dataset_labelled = get_dummy(
            target_dataset, excerpt, pseudo_labels, get_dataset=True)

        # re-merge soruce data and target data
        merged_dataset = ConcatDataset([source_dataset,
                                        target_dataset_labelled])

        # save model
        if ((k + 1) % cfg.save_step == 0):
            save_model(F, "adapt-F-{}.pt".format(k + 1))
            save_model(F_1, "adapt-F_1-{}.pt".format(k + 1))
            save_model(F_2, "adapt-F_2-{}.pt".format(k + 1))
            save_model(F_t, "adapt-F_t-{}.pt".format(k + 1))

    # save final model
    save_model(F, "adapt-F-final.pt")
    save_model(F_1, "adapt-F_1-final.pt")
    save_model(F_2, "adapt-F_2-final.pt")
    save_model(F_t, "adapt-F_t-final.pt")
