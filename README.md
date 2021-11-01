# DetMS

This repo contains the codes for the project *detecting mixing services in bitcoin transactions via deep learning*.

**Keywords**: graph node classification, node embedding, dynamic network, temporal random walk, skip-gram, neural network

## Original paper and dataset
\[Wu et al.'20\] *Detecting Mixing Services via Mining Bitcoin Transaction Network with Hybrid Motifs*, IEEE Trans. on Systems, Man, and Cybernetics, 2020.
Dataset avaliable at http://xblock.pro/bitcoin/#BMD. Datasets and the core idea of the project are referred to, but we do NOT adopt the methodology in this paper.

## Method
We build a temporal network `G(V,E)` where `V={addrs}` and `{weight, time}` is the attribute of each edge in `E`. Node features are extracted via a time-respecting adaptation of `node2vec`. Refer to \[Wen et al.'20\] *Bitcoin Transaction Forecasting with Deep Network Representation Learning*, though the details may vary. 

Then the features are fed into classifiers for classification task. We consider using LR, SVM, DNN, or LSTM.
