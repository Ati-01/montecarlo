"""
Unit and regression test for the montecarlo package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import montecarlo

import numpy as np
import networkx as nx

def test_montecarlo_IsingHamiltonian():
    N = 6
    Jval = 2.0
    G = nx.Graph()
    G.add_nodes_from([i for i in range(N)])
    G.add_edges_from([(i,(i+1)% G.number_of_nodes() ) for i in range(N)])
    for e in G.edges:
        G.edges[e]['weight'] = Jval
    
    mus = np.zeros(len(G.nodes()))
    J = [[] for i in G.nodes()]
    for e in G.edges:
        J[e[0]].append((e[1], G.edges[e]['weight']))
        J[e[1]].append((e[0], G.edges[e]['weight']))
    
    conf = montecarlo.BitString(N)
    ham = montecarlo.IsingHamiltonian(J,mus)
    E, M, HC, MS = ham.compute_average_values(1)

    assert(np.isclose(E,  -11.95991923))
    assert(np.isclose(M,   -0.00000000))
    assert(np.isclose(HC,   0.31925472))
    assert(np.isclose(MS,   0.01202961))

def test_montecarlo_IsingHamiltonian_get_lowest_energy_config():
    mus = [0.0 for i in range(3)]
    J = [[] for i in range(3)]
    for site in range(2):
        J[site].append((site+1, 1))
    ham = montecarlo.IsingHamiltonian(J,mus)
    ham.mu[0] = 1.2
    emin, cmin = ham.get_lowest_energy_config()
    test = [0, 1, 0]

    assert(np.isclose(emin,   -3.20000))
    assert((cmin == test).all())

def test_bitstring_constructor():
    bs = montecarlo.BitString(4)
    test = np.array([0, 0, 0, 0])

    assert((bs.config == test).all())
    assert(bs.N == 4)
    assert(len(bs) == 4)
    
def test_bitstring_equality():
    bs1 = montecarlo.BitString(4)
    bs1.config = np.array([1, 0, 1, 0])
    bs2 = montecarlo.BitString(4)
    bs2.config = np.array([1, 0, 1, 0])
    bs3 = montecarlo.BitString(4)
    bs3.config = np.array([1, 1, 1, 0])

    assert(bs1 == bs2)
    assert(bs1 != bs3)
    assert(bs2 != bs3)

def test_bitstring_str():
    bs = montecarlo.BitString(4)
    bs.config = np.array([1, 0, 1, 0])

    assert(str(bs) == "[ 1 0 1 0 ]")


def test_bitstring_set_config():
    bs = montecarlo.BitString(5)
    test = np.array([1, 0, 0, 1, 0])
    bs.set_config([1, 0, 0, 1, 0])

    assert((bs.config == test).all())

def test_bitstring_set_int_config():
    bs = montecarlo.BitString(8)
    bs.set_int_config(97)
    test = np.array([0, 1, 1, 0, 0, 0, 0, 1])

    assert((bs.config == test).all())

def test_bitstring_int():
    bs1 = montecarlo.BitString(8)
    bs1.config = np.array([0, 1, 1, 0, 0, 0, 0, 1])
    bs2 = montecarlo.BitString(8)

    assert((bs1.int() == 97))
    assert((bs2.int() == 0))

def test_bitstring_flip_site():
    bs = montecarlo.BitString(8)
    bs.config = np.array([0, 1, 1, 0, 0, 0, 0, 1])
    bs.flip_site(7)
    bs.flip_site(6)
    test = np.array([0, 1, 1, 0, 0, 0, 1, 0])

    assert((bs.config == test).all())

def test_bitstring_on():
    bs = montecarlo.BitString(8)
    bs.config = np.array([0, 1, 1, 0, 0, 0, 0, 1])

    assert((bs.on() == 3))

def test_bitstring_off():
    bs = montecarlo.BitString(8)
    bs.config = np.array([0, 1, 1, 0, 0, 0, 0, 1])

    assert((bs.off() == 5))