import gzip
import numpy as np

from matplotlib import cm
from matplotlib import colors
from scipy.spatial.distance import pdist, squareform


def unzip_res_range(res_range):
    """
    Similar to `unzip_res_range` in mjm_tools.
    """
    if res_range == '':
        return []
    residue_list = []
    chars_list = res_range[1:-1].split(',')
    for chars in chars_list:
        if '-' in chars:
            start, end = chars.split('-')
            residue_list.extend(list(range(int(start), int(end) + 1)))
        else:
            residue_list.append(int(chars))
    return residue_list


def parse_pdb_coords(model_file):
    """
    Parse residue coordinates and residue numbers from a gzipped PDB file.

    Args:
      model_file: str, full path to the model file.

    Returns:
      A 2-D array of shape [num_residues, 3] containing coordinates of the CA atom of all residues
      and an 1-D array of residue numbers.
    """
    coords = []
    resnums = []
    with gzip.open(model_file, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            if not line.startswith('ATOM '):
                continue
            if line[12:16].strip() != 'CA':
                continue
            resnum = line[22:26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append(np.array([x,y,z]))
            resnums.append(resnum)
    return np.array(coords), np.array(resnums)


def calculate_pairwise_dist(model_file):
    """
    Calculate the pairwise distance matrix for residues in a PDB chain.

    Args:
      model_file:  str, full path to the model file.

    Returns:
      A tuple of two arrays, the first being the pairwise distance matrix in vector form, and the second being
      residue numbers in the PDB chain.
    """
    coords, resnums = parse_pdb_coords(model_file)
    pairwise_dist = pdist(coords)
    return pairwise_dist, resnums


def generate_spatial_weight(pairwise_dist, wtype='inverse'):
    """
    Generate the weight matrix for spatial statistics.

    Args:
      pairwise_dist: np.array, pairwise distance matrix in flat form.
      wtype:         str, the type of weight, only supports `inverse` and `inverse_squared`.

    Returns:
      A weight matrix in square form.

    Raises:
      ValueError: if the weight type is not supported.
    """
    if wtype == 'inverse':
        return squareform(1 / pairwise_dist)
    elif wtype == 'inverse_squared':
        return squareform(1 / (pairwise_dist * pairwise_dist))
    else:
        raise ValueError('Weight type %s not supported.' % wtype)


def getis_ord_G(score_array, weight_mat):
    """
    Calculate local Getis and Ord's G.

    Args:
      score_array: np.array, the score corresponding to each point.
      weight_mat:  np.array, weight matrix in square form.

    Returns:
      An array of z-scores corresponding to each point.
    """
    n = score_array.shape[0]
    x_bar = np.mean(score_array)
    s = np.sqrt(np.mean(score_array * score_array) - x_bar * x_bar)
    numerator = np.sum(np.matmul(weight_mat, (score_array - x_bar).reshape((-1, 1))), axis=1)
    denominator = s * np.sqrt((n * np.sum(weight_mat * weight_mat, axis=1) - np.sum(weight_mat, axis=1) ** 2) / (n - 1))
    z_array = numerator / denominator
    return z_array


def calc_g_from_model(scores, model_file, uniprot_res, weight='inverse'):
    """
    Calculate Getis and Ord's G for a single protein.

    Args:
      scores:      list or numpy.array, array of raw scores.
      model_file:  str, full path to the model file.
      uniprot_res: str, residue range on the UniProt basis.
      weight:      str, the type of weight for calculating pairwise matrix,
                   only supports `inverse` and `inverse_squared`.

    Returns:
      An array of z-scores corresponding to each residue.
    """
    pairwise_dist, resnums = calculate_pairwise_dist(model_file)
    keep_idx = np.where(np.isin(resnums, np.array(unzip_res_range(uniprot_res))))
    pairwise_dist = squareform(squareform(pairwise_dist)[keep_idx[0]][:, keep_idx[0]])
    resnums = np.array([int(r) for r in resnums[keep_idx[0]]])
    score_array = np.array(scores)[resnums.astype(int) - 1]
    z_array = getis_ord_G(score_array, generate_spatial_weight(pairwise_dist, weight))
    return z_array, resnums


def generate_colors(values, cmap_name='', cmap=None, reverse=False):
    """
    Generates an array of colors according to `values`.

    Args:
      values:    np.array, an array of number values.
      cmap_name: str, name of the colormap.
      cmap:      matplotlib.colors.Colormap, a colormap provided.
      reverse:   bool, whether to reverse the colormap.

    Returns:
      An 2-D np.array with width 3, specifying the color in RGB format corresponding to each value
      in `values`.

    Raises:
      ValueError: if the colormap provided is invalid.
    """
    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    if reverse:
        values = 1.0 - values
    if cmap_name:
        cmap = cm.get_cmap(cmap_name)
    if cmap is None:
        raise ValueError('Invalid color map.')
    color_array = np.array([cmap(v)[:3] for v in values])
    return [colors.to_hex(x) for x in color_array]
