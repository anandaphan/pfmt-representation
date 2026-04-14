import numpy as np
import utils
import torch
import torch.nn.functional as F

EPSILON = 1e-20

def threshold_decoder(nNotes, vec):
    # nNotes 1 : 0-4
    # nNotes 2 : 5-14
    # nNotes 3 : 15-24
    # nNotes 4 : 25-29
    # nNotes 5 : 30
    key = np.empty_like(vec)
    key[(nNotes==1) & (vec<=0.20)] = 0
    key[(nNotes==1) & (vec>0.20) & (vec<=0.40)] = 1
    key[(nNotes==1) & (vec>0.40) & (vec<=0.60)] = 2
    key[(nNotes==1) & (vec>0.60) & (vec<=0.80)] = 3
    key[(nNotes==1) & (vec>0.80) & (vec<=1)] = 4

    key[(nNotes==2) & (vec<=0.1)] = 5
    key[(nNotes==2) & (vec>0.10) & (vec<=0.20)] = 6
    key[(nNotes==2) & (vec>0.20) & (vec<=0.30)] = 7
    key[(nNotes==2) & (vec>0.30) & (vec<=0.40)] = 8
    key[(nNotes==2) & (vec>0.40) & (vec<=0.50)] = 9
    key[(nNotes==2) & (vec>0.50) & (vec<=0.60)] = 10
    key[(nNotes==2) & (vec>0.60) & (vec<=0.70)] = 11
    key[(nNotes==2) & (vec>0.70) & (vec<=0.80)] = 12
    key[(nNotes==2) & (vec>0.80) & (vec<=0.90)] = 13
    key[(nNotes==2) & (vec>0.90) & (vec<=1)] = 14

    key[(nNotes==3) & (vec<=0.1)] = 15
    key[(nNotes==3) & (vec>0.10) & (vec<=0.20)] = 16
    key[(nNotes==3) & (vec>0.20) & (vec<=0.30)] = 17
    key[(nNotes==3) & (vec>0.30) & (vec<=0.40)] = 18
    key[(nNotes==3) & (vec>0.40) & (vec<=0.50)] = 19
    key[(nNotes==3) & (vec>0.50) & (vec<=0.60)] = 20
    key[(nNotes==3) & (vec>0.60) & (vec<=0.70)] = 21
    key[(nNotes==3) & (vec>0.70) & (vec<=0.80)] = 22
    key[(nNotes==3) & (vec>0.80) & (vec<=0.90)] = 23
    key[(nNotes==3) & (vec>0.90) & (vec<=1)] = 24
    
    key[(nNotes==4) & (vec<=0.20)] = 25
    key[(nNotes==4) & (vec>0.20) & (vec<=0.40)] = 26
    key[(nNotes==4) & (vec>0.40) & (vec<=0.60)] = 27
    key[(nNotes==4) & (vec>0.60) & (vec<=0.80)] = 28
    key[(nNotes==4) & (vec>0.80) & (vec<=1)] = 29
    
    key[(nNotes==5)] = 30
    return key, nNotes

def topk_sample_to_keys(nNotes, vec):
    key = np.empty_like(nNotes)
    for i, k in enumerate(nNotes):
        key[i] = utils.finger2key[tuple(np.sort(vec[i,-k:]))]
    return key

def priority_based_decoder(nNotes, vec):
    # each chromosome as the score of the finger
    # Direct top-k
    vec_ = vec.reshape(len(nNotes),5)
    vec_ = np.argsort(vec_,axis=1)
    key = topk_sample_to_keys(nNotes, vec_)
    return key, nNotes

def controlled_softmax_decoder(nNotes, vec, tau):
    vec_ = vec.reshape(len(nNotes),5)

    vec_torch = torch.from_numpy(vec_)
    vec_softmax = F.softmax((vec_torch+EPSILON)/tau, dim=1) 
    vec_softmax = np.argsort(vec_softmax.numpy(),axis=1)
    key = topk_sample_to_keys(nNotes, vec_softmax)
    return key, nNotes

def k_block_decoder(nNotes, vec):
    key = np.empty_like(nNotes)
    offset = np.cumsum(nNotes)[:-1]
    vec_parts = np.split(vec, offset)
    finger = np.array([0,1,2,3,4])
    for i, vec_part in enumerate(vec_parts):
        decoded = []
        temp = finger.tolist()
        for j in range(len(vec_part)):
            position = int(np.clip(np.floor(vec_part[j] * i), 0, len(temp)-1))
            decoded.append(temp[position])
            temp.pop(position)
        key[i] = utils.finger2key[tuple(set(decoded))]
    return key, nNotes


def relaxation_with_threshold(nNotes, vec, threshold = 0.5):
    nNotesDecoded = np.zeros_like(nNotes)
    vec_ = vec.reshape(len(nNotes),5)
    vec_argsort = np.argsort(vec_,axis=1)
    key = np.empty_like(nNotes)

    # relaxation with threshold
    vec_chosen = np.where(vec_>threshold, 1, 0)
    for i, k in enumerate(nNotes):
        nNotesDecoded[i] = vec_chosen[i].sum()
        if vec_chosen[i].sum() == k:
            key[i] = utils.finger2key[tuple(np.where(vec_chosen[i]==1)[0])]
        else:
            key[i] = utils.finger2key[tuple(np.sort(vec_argsort[i,-k:]))]
    return key, nNotesDecoded


def decoder_latent(nNotes,z):
    z_ = z.reshape(len(nNotes),2)
    key = np.empty_like(nNotes)
    nNotesDecoded = np.zeros_like(nNotes)   


    return key, nNotesDecoded
    
    