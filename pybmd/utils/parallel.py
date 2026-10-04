'''
Module implementing the small set of parallel utilities used across the
library.  Every function degrades to a no-op or the identity when ``comm`` is
None, so serial and parallel code paths are the same code.

PyBMD replicates the data on every rank and distributes the *triad* loop, so
none of the collective-I/O machinery a domain-decomposed solver would need
appears here.
'''
import numpy as np


def _get_module_MPI(comm):
    '''Get the MPI module from the communicator's own package.'''
    prefix = type(comm).__module__.split('.', 1)[0]
    MPI = __import__(f'{prefix}.MPI', fromlist=[None])
    return MPI


def pr0(string, comm):
    '''
    Print on rank 0 only.

    :param str string: what to print.
    :param MPI.Comm comm: parallel communicator, or None.
    '''
    if comm is None or comm.rank == 0:
        print(string)


def barrier(comm):
    '''
    Synchronize all ranks.

    :param MPI.Comm comm: parallel communicator, or None.
    '''
    if comm is not None:
        comm.Barrier()


def allreduce(data, comm):
    '''
    Sum an array across all ranks.

    :param numpy.ndarray data: local contribution.
    :param MPI.Comm comm: parallel communicator, or None.

    :return: the sum over all ranks, identical on every rank.
    :rtype: numpy.ndarray
    '''
    if comm is None:
        return data
    MPI = _get_module_MPI(comm)
    # MPI needs a contiguous native-endian buffer; astype/ascontiguousarray
    # convert the values, where a .view() would only reinterpret the bytes
    data = np.ascontiguousarray(data, dtype=data.dtype.newbyteorder('='))
    reduced = np.zeros_like(data)
    comm.Barrier()
    comm.Allreduce(data, reduced, op=MPI.SUM)
    return reduced


def distribute_indices(n, comm):
    '''
    Split ``range(n)`` across ranks, round-robin.

    :param int n: number of items, here the number of triads.
    :param MPI.Comm comm: parallel communicator, or None.

    :return: the indices owned by this rank.
    :rtype: numpy.ndarray

    .. note::

        Round-robin rather than contiguous blocks because the cost of a triad
        varies systematically across the ``f1``-``f2`` plane: the
        numerical-radius solve takes more iterations where the spectrum of
        ``B`` is clustered, which happens in bands. A contiguous split would
        hand one rank an entire band; interleaving balances the load with an
        imbalance of at most one triad.
    '''
    if comm is None:
        return np.arange(n)
    return np.arange(comm.rank, n, comm.size)
