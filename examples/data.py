#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Loader for the cylinder-wake dataset used by the examples.'''
import os
import numpy as np


def load_cylinder_wake(PATH):
    '''
    Load the cylinder wake at Re=500.

    :param bool prefer_full: try the full dataset first. Default is True.

    :return: ``x``, ``y``, ``u``, ``v``, ``dt``.
    :rtype: tuple
    '''
    with np.load(PATH) as d:
        return (d['x'].astype(np.float64), d['y'].astype(np.float64),
                d['u'].astype(np.float64), d['v'].astype(np.float64),
                float(d['dt']))
