"""Test script for ATDA."""

import torch.nn as nn

from misc.utils import make_variable


def evaluate(encoder, classifier, data_loader):
    """Evaluate classifier on source or target domains."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels.squeeze_())

        #print('images shape = {}'.format(images.shape))
        #print('encoding shape = {}'.format(encoder(images).shape))
        preds = classifier(encoder(images))
        #print('preds shape = {}'.format(preds.shape))
        #print('labels shape = {}'.format(labels.shape))
        cv = criterion(preds, labels)
        #print('crit val = {}'.format(cv))
        loss += cv.item() #data[0]

        pred_cls = preds.data.max(1)[1]
        #print('labels.data = {}, preds_cls = {}'.format(labels.data, pred_cls)) #!!
        #print('eq = {}'.format(pred_cls.eq(labels.data)))
        contrib =  pred_cls.eq(labels.data).cpu().sum()
        #print('contrib = {}'.format(contrib))
        acc += float(contrib)

    loss /= float(len(data_loader))
    acc /= float(len(data_loader.dataset))

    print("Avg Loss = {:.5f}, Avg Accuracy = {:2.5%}".format(loss, acc))
