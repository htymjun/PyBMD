'''
Module implementing the small set of parallel utilities used across the
library.  Every function degrades to a no-op or the identity when ``comm`` is
None, so serial and parallel code paths are the same code.

PyBMD distributes the *triad* loop across all ranks. The large arrays (data,
Fourier realizations, stored modes) are not replicated per rank but held once
per node in MPI-3 shared memory, which the ranks of a node read directly; the
DFT is split across the ranks of a node by spatial point. No collective I/O is
needed: the data is read by the first rank of each node.
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
        print(string, flush=True)


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


def node_comm(comm):
    '''
    The sub-communicator of the ranks that share memory with this one.

    :param MPI.Comm comm: parallel communicator, or None.

    :return: the node communicator, or None.
    '''
    if comm is None:
        return None
    MPI = _get_module_MPI(comm)
    return comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank)


def shared_empty(shape, dtype, comm):
    '''
    An uninitialized array held once in the shared memory of ``comm``.

    The first rank of ``comm`` allocates the memory and every rank maps it, so
    a write by any rank is seen by all of them after a barrier.

    :param tuple shape: shape of the array.
    :param dtype: data type of the array.
    :param MPI.Comm comm: node communicator from :func:`node_comm`, or None.

    :return: ``(array, win)``; ``win`` is the MPI window backing the array,
        to be released with :func:`free_shared` once no rank uses the array,
        or None for a private array when ``comm`` is None.
    :rtype: tuple(numpy.ndarray, MPI.Win)
    '''
    dtype = np.dtype(dtype)
    if comm is None:
        return np.empty(shape, dtype=dtype), None
    MPI = _get_module_MPI(comm)
    nbytes = int(np.prod(shape)) * dtype.itemsize if comm.rank == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, dtype.itemsize, comm=comm)
    buf, _ = win.Shared_query(0)
    return np.ndarray(shape, dtype=dtype, buffer=buf), win


def shared_zeros(shape, dtype, comm):
    '''As :func:`shared_empty`, zero-filled.'''
    arr, win = shared_empty(shape, dtype, comm)
    if comm is None or comm.rank == 0:
        arr.fill(0)
    barrier(comm)
    return arr, win


def free_shared(win, comm):
    '''
    Release a window from :func:`shared_empty` on every rank of ``comm``.
    Arrays mapping it must not be used afterwards.

    :param MPI.Win win: the window, or None.
    :param MPI.Comm comm: the node communicator it was allocated on.
    '''
    if win is None:
        return
    barrier(comm)
    win.Free()


def allreduce_across_nodes(data, comm, node):
    '''
    Sum a shared array in place across nodes; within a node it is a single
    array already, so only the first rank of each node takes part. Each entry
    must be written on exactly one node, so the sum adds exact zeros.

    :param numpy.ndarray data: the shared array, from :func:`shared_zeros`.
    :param MPI.Comm comm: parallel communicator, or None.
    :param MPI.Comm node: its node communicator, from :func:`node_comm`.
    '''
    if comm is None:
        return
    barrier(node)
    if node.size != comm.size:
        MPI = _get_module_MPI(comm)
        leaders = comm.Split(0 if node.rank == 0 else MPI.UNDEFINED,
                             comm.rank)
        if leaders != MPI.COMM_NULL:
            leaders.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)
            leaders.Free()
    barrier(node)


def split_range(n, comm):
    '''
    Split ``range(n)`` into contiguous, near-equal pieces, one per rank.

    Contiguous rather than round-robin, unlike :func:`distribute_indices`:
    used for the spatial points of the DFT, whose cost is uniform, where a
    contiguous piece keeps memory access local.

    :param int n: number of items.
    :param MPI.Comm comm: parallel communicator, or None.

    :return: ``(start, stop)`` of this rank's piece.
    :rtype: tuple(int, int)
    '''
    if comm is None:
        return 0, n
    base, rem = divmod(n, comm.size)
    start = comm.rank * base + min(comm.rank, rem)
    return start, start + base + (comm.rank < rem)
