import zarr
import numpy as np


def savez_zarr(path, append=False, chunks=None, **arrays):
    """
    Drop-in replacement for np.savez / np.savez_compressed using Zarr.

    Parameters
    ----------
    path : str
        Output path ('.zarr' will be added if missing)
    append : bool
        If True, append along axis 0 if dataset exists
    chunks : tuple or "auto"
        Zarr chunking
    **arrays : dict
        Arrays to store
    """

    if not path.endswith(".zarr"):
        path += ".zarr"

    mode = "a" if append else "w"
    z = zarr.open_group(path, mode=mode)

    for name, arr in arrays.items():
        arr = np.asarray(arr)

        if name in z and append:
            z[name].append(arr)
        else:
            maxshape = (None,) + arr.shape[1:]

            z.create_dataset(
                name,
                data=arr,
                chunks=chunks,
                maxshape=maxshape,
                overwrite=True
            )