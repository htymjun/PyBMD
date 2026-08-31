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
    data = data.view(data.dtype.newbyteorder('='))
    reduced = np.zeros_like(data)
    comm.Barrier()
    comm.Allreduce(data, reduced, op=MPI.SUM)
    return reduced


def allreduce_scalar(value, comm, op='sum'):
    '''
    Reduce a scalar across all ranks.

    :param value: local value.
    :param MPI.Comm comm: parallel communicator, or None.
    :param str op: one of 'sum', 'min', 'max'. Default is 'sum'.

    :return: the reduced value, identical on every rank.
    '''
    if comm is None:
        return value
    MPI = _get_module_MPI(comm)
    ops = {'sum': MPI.SUM, 'min': MPI.MIN, 'max': MPI.MAX}
    return comm.allreduce(value, op=ops[op])


def _blockdist(n, size, rank):
    '''Contiguous block distribution of ``n`` items; returns (count, start).'''
    q, r = divmod(n, size)
    count = q + (1 if r > rank else 0)
    start = rank * q + min(rank, r)
    return (count, start) if rank < size else (0, 0)


def distribute_indices(n, comm, mode='round_robin'):
    '''
    Split ``range(n)`` across ranks.

    :param int n: number of items, here the number of triads.
    :param MPI.Comm comm: parallel communicator, or None.
    :param str mode: 'round_robin' (default) or 'block'.

    :return: the indices owned by this rank.
    :rtype: numpy.ndarray

    .. note::

        The default is round-robin rather than contiguous blocks because the
        cost of a triad varies systematically across the ``f1``-``f2`` plane:
        the numerical-radius solve takes more iterations where the spectrum of
        ``B`` is clustered, which happens in bands. A contiguous split would
        hand one rank an entire band; interleaving balances the load with an
        imbalance of at most one triad.
    '''
    if comm is None:
        return np.arange(n)
    if mode == 'round_robin':
        return np.arange(comm.rank, n, comm.size)
    if mode == 'block':
        count, start = _blockdist(n, comm.size, comm.rank)
        return np.arange(start, start + count)
    raise ValueError(f"mode must be 'round_robin' or 'block'; got {mode!r}.")
