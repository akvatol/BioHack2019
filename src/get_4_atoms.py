import h5py
import argparse as ag
from sqlitedict import SqliteDict
from itertools import product


def getArgs():

    parser = ag.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input-file',
        required=True,
        dest='inp',
        type=str,
        help='Input file with peptides')
    parser.add_argument(
        '-sqld',
        '--database-file',
        required=True,
        dest='sqld',
        type=str,
        help='Input sqlitedict database')
    parser.add_argument(
        '-o',
        '--output-file',
        required=True,
        dest='out',
        type=str,
        help='Out file name')
    parser.add_argument(
        '-k',
        '--k in k-mers',
        required=True,
        dest='k',
        type=int,
        help='Length of peptides')

    args = vars(parser.parse_args())
    return args


def get_indices(pep_names, db_path, k=3):
    amino_alphabet = [
        'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'S', 'T', 'N', 'Q', 'C', 'G',
        'P', 'R', 'H', 'K', 'D', 'E'
    ]
    db = SqliteDict(db_path)
    pep_names = [''.join(kmer) for kmer in product(amino_alphabet, repeat=k)]
    dic = {
        pep: db[pep].select('resnum 2:5 and name N CA C O').getIndices()
        for pep in pep_names
    }
    return dic


def write_to_hdf5(write_path, old_data, name_dic):
    with h5py.File(write_path, 'w') as new_data:
        for k in old_data.keys():
            if k in name_dic.keys():
                temp = new_data.create_group(k)
                conformations = old_data[k].keys()
                for conf in conformations:
                    temp.create_dataset(
                        conf, data=old_data[k][conf][name_dic[k], :])
            else:
                continue


if __name__ == '__main__':

    args = getArgs()

    data_file = h5py.File(args['inp'])

    name_dic = get_indices(data_file.keys(), args['sqld'], k=args['k'])

    write_to_hdf5(args['out'], data_file, name_dic)
