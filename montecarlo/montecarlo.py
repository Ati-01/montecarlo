"""Provide the primary functions."""

import math
import numpy as np

class BitString:
    """Simple class to implement a config of bits"""
    def __init__(self, N):
        """Constructor

        Parameters
        ----------
        N: integer, length of bitstring
        """
        self.N = N
        self.config = np.zeros(N, dtype=int) 

    def __str__(self):
        """When bitstring is converted to a string, it lists the bits."""
        string = '[ '
        for x in self.config:
            string = string + str(x) + ' '
        string = string + ']'
        return string

    def __eq__(self, other):
        """When two bitstrings are being compared, it checks to see if the elements are the same"""
        for x in range(self.N):
            if self.config[x] != other.config[x]:
                  return False
        return True
    
    def __len__(self):
        """When len() is called on a bitstring, it returns the number of bits in the bitstring"""
        return self.N

    def on(self):
        """Find the number of 1 bits

        Returns
        -------
        ones  : int
            Number of ones in bitstring
        """
        ones = 0
        for x in self.config:
            if x == 1:
                ones = ones + 1
        return ones
    
    def off(self):
        """Find the number of 0 bits

        Returns
        -------
        zeroes  : int
            Number of zeroes in bitstring
        """
        zeroes = 0
        for x in self.config:
            if x == 0:
                zeroes = zeroes + 1
        return zeroes
    
    def flip_site(self,i):
        """Flips the bit at the specified element

        Parameters
        ----------
        i   : int
            ith element in bitstring
        """
        if self.config[i] == 1:
            self.config[i] = 0
        else:
            self.config[i] = 1

    
    def int(self):
        """Gives base 10 number representation of bitstring

        Returns
        -------
        dec  : int
            Decimal value of bitstring
        """
        dec = 0
        for x in range(self.N):
            if self.config[x] == 1:
                dec = dec + 2**(self.N - x - 1)
        return dec
 

    def set_config(self, s:list[int]):
        """Sets bitstring to a given bitstring

        Parameters
        ----------
        s   : List of integers
            List of bitstrings
        """
        for x in range(self.N):
            self.config[x] = s[x]
        
    def set_int_config(self, dec:int):
        """Sets bitstring to a given decimal number

        Parameters
        ----------
        dec   : int
            Decimal value to change bitstring to
        """
        other_bs = BitString(self.N)
        for x in range(other_bs.N):
            if dec == 0:
                break
            if math.log2(dec) >= other_bs.N - x - 1:
                other_bs.config[x] = 1
                dec = dec - 2**(other_bs.N - x - 1)
        self.config = other_bs.config

class IsingHamiltonian:
    """Class for an Ising Hamiltonian of arbitrary dimensionality

    .. math::
        H = \\sum_{\\left<ij\\right>} J_{ij}\\sigma_i\\sigma_j + \\sum_i\\mu_i\\sigma_i

    """

    def __init__(self, J=[[()]], mu=np.zeros(1)):
        """Constructor

        Parameters
        ----------
        J: list of lists of tuples, optional
            Strength of coupling, e.g,
            [(4, -1.1), (6, -.1)]
            [(5, -1.1), (7, -.1)]
        mu: vector, optional
            local fields
        """
        self.J = J
        self.mu = mu

        self.nodes = []
        self.js = []

        for i in range(len(self.J)):
            self.nodes.append(np.zeros(len(self.J[i]), dtype=int))
            self.js.append(np.zeros(len(self.J[i])))
            for jidx, j in enumerate(self.J[i]):
                self.nodes[i][jidx] = j[0]
                self.js[i][jidx] = j[1]
        self.mu = np.array([i for i in self.mu])
        self.N = len(self.J)

    def energy(self, config):
        """Compute energy of configuration, `config`

            .. math::
                E = \\left<\\hat{H}\\right>

        Parameters
        ----------
        config   : BitString
            input configuration

        Returns
        -------
        energy  : float
            Energy of the input configuration
        """
        if len(config.config) != len(self.J):
            error("wrong dimension")

        e = 0.0
        for i in range(config.N):
            # print()
            # print(i)
            for j in self.J[i]:
                if j[0] < i:
                    continue
                # print(j)
                if config.config[i] == config.config[j[0]]:
                    e += j[1]
                else:
                    e -= j[1]

        e += np.dot(self.mu, 2 * config.config - 1)
        return e

    def compute_average_values(self, T):
        """Compute Average values exactly

        Parameters
        ----------
        T      : int
            Temperature

        Returns
        -------
        E  : float
            Energy
        M  : float
            Magnetization
        HC : float
            Heat Capacity
        MS : float
            Magnetic Susceptability
        """
        E = 0.0
        M = 0.0
        Z = 0.0
        EE = 0.0
        MM = 0.0

        conf = BitString(self.N)

        for i in range(2 ** conf.N):
            conf.set_int_config(i)
            Ei = self.energy(conf)
            Zi = np.exp(-Ei / T)
            E += Ei * Zi
            EE += Ei * Ei * Zi
            Mi = np.sum(2 * conf.config - 1)
            M += Mi * Zi
            MM += Mi * Mi * Zi
            Z += Zi

        E = E / Z
        M = M / Z
        EE = EE / Z
        MM = MM / Z

        HC = (EE - E * E) / (T * T)
        MS = (MM - M * M) / T
        return E, M, HC, MS

    def get_lowest_energy_config(self):
        """Finds lowest energy configuration

        Parameters
        ----------
        qubits   : Bitstring
            input configuration
        G    : Graph
            input graph defining the Hamiltonian
        Returns
        -------
        emin  : float
            Lowest energy
        my_bs.config  : Bitstring
            Bitstring configuration with lowest energy
        """
        x = [] # Store list of indices
        y = [] # Store list of energies
        xmin = None # configuration of minimum energy configuration
        emin = 0 # minimum of energy
        my_bs = BitString(self.N)

        my_bs.set_int_config(0)
        emin = self.energy(my_bs)
        for i in range(np.power(2, my_bs.N)):
            my_bs.set_int_config(i)
            ecurrent = self.energy(my_bs)
            x.append(i)
            y.append(ecurrent)
            if ecurrent < emin:
                emin = ecurrent
                xmin = i
        my_bs.set_int_config(xmin)
        return emin, my_bs.config

def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
