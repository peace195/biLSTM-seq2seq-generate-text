#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os

def embed_w2v(embedding, data_set):
    embedded = [map(lambda x: embedding[x], sample) for sample in data_set]
    return embedded


def apply_sparse(data_set):
    applied = [map(lambda x: [x], sample) for sample in data_set]
    return applied


def pad_to(lst, length, value):
    for i in range(len(lst), length):
        lst.append(value)
    
    return lst


def uprint(x):
    print(repr(x).decode('unicode-escape'))


def uprintln(x):
    print(repr(x).decode('unicode-escape'))

