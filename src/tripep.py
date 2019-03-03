# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from itertools import product
import h5py
from tqdm import tqdm
from mpi4py import MPI
from numba import jit
import argparse as ag
"""
Для чего пакет
"""

TRASHHOLD = 0.5

amino_alphabet = [
    'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'S', 'T', 'N', 'Q', 'C', 'G', 'P',
    'R', 'H', 'K', 'D', 'E'
]


def time(func):
    import time

    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        res = func(*args, **kwargs)
        print(func.__name__, time.perf_counter() - t)
        return res

    return wrapper


def get_pep(k=None):
    tripep = [''.join(kmer) for kmer in product(amino_alphabet, repeat=k)]
    return tripep


@jit(parallel=True)
def walker(pep_type: str):
    """
    Генерирует все возможные пептиды, которые можно соеденить с 
    pep_type
    """
    pep_list = []
    for j in range(0, len(get_pep(len(pep_type) - 1))):
        pos_pep = pep_type[0] + get_pep(len(pep_type) - 1)[j]
        pep_list.append(pos_pep)
        j += 1
    cutted_pep_list = set(pep_list)
    cutted_pep_list.remove(pep_type)
    return tuple(cutted_pep_list)


def pair_creator(first_pep: str, second_pep: str):
    """
    Принимает на вход два пептида и возвращает все возможные пары
    комбинаций с помощью которых их можно соединить
    """
    all_pairs = []
    min_intersec = min((len(first_pep), len(second_pep)))

    for i in range(1, min_intersec):
        if second_pep.endswith(first_pep[:i]):
            all_pairs.append(''.join([str(x) for x in range(i)]) + ''.join([
                str(x)
                for x in reversed(range(len(second_pep) - i, len(second_pep)))
            ]))

        # if first_pep.endswith(second_pep[:i]):
        #     all_pairs.append(''.join([
        #         str(x)
        #         for x in reversed(range(len(first_pep) - i, len(first_pep)))
        #     ]) + ''.join([str(x) for x in range(i)]))

    return all_pairs


def mean(numbers):
    """
    Функция для расчета среднего арифметического
    """
    return float(sum(numbers)) / max(len(numbers), 1)


def chunkit(data: list or tuple, n=None):
    """
    Функция разбивает исходный массив на N частей (N == n).

    Arguments
    ---------
    data_list: list or tuple
        Массив, который будет разделен на n частей
    n: int
        Число подмассивов в возвращаемом массиве (default: 2)

    Returns
    -------
    list: разделенный на части список

    Example
    -------
    >>> l = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> a = chunkit(l)
    >>> print(a)
    [[1, 2, 3, 4], [5, 6, 7, 8]]
    >>> b = chunkit(l, n=4)
    >>> print(b)
    [[1, 2], [3, 4], [5, 6], [7, 8]]
    """
    new_data = []
    if not n:
        n = 2
    avg = len(data) / n
    last = 0
    while last < len(data):
        new_data.append(data[int(last):int(last + avg)])
        last += avg
    return np.array(new_data)


def rmsd_calc(coord1, coord2):
    """
    Считает RMSD для пар координат.
    """
    return np.sqrt(mean_squared_error(coord1, coord2))


class Tripep:

    __slots__ = ('peptides', 'docking_pose', 'coordinates')

    def __init__(self, name: str, conf: str, coordinates: np.array):
        """
        peptides: str
            Строка с последовательность аминокислот

        conformation: str (Number)
            Номер конформера (От 0 до 9)

        coordinates: np.array
            Матрица координатами (N Ca C O) атомов аминокислот
        """
        self.peptides = name
        self.docking_pose = conf
        self.coordinates = chunkit(data=coordinates, n=len(name))

    def comparsion_uni(self, obj):
        """
        Принимает другой объект Tripep, и возвращает список с
        Сотавом 1 пептида, позу докинга 1, состав пептида 2, номера
        аминокислот которыми застыковались пептиды
        """
        pairs = pair_creator(self.peptides, obj.peptides)
        if pairs:
            for pair in pairs:
                rmsd_list = []
                N_d_list = []
                vec_d_list = []
                aminoacids = zip(pair[:len(pair) // 2], pair[len(pair) // 2:])
                for aminoacid1, aminoacid2 in aminoacids:
                    rmsd_list.append(
                        self._comparsion(aminoacid1, aminoacid2, obj))
                    N_d_list.append(self.N_d(aminoacid1, aminoacid2, obj))
                    vec_d_list.append(self.vec_d(aminoacid1, aminoacid2, obj))

                if all(x < TRASHHOLD for x in rmsd_list) and all(
                        x < TRASHHOLD
                        for x in N_d_list) and all(x < TRASHHOLD
                                                   for x in vec_d_list):
                    return (self.peptides, self.docking_pose, obj.peptides,
                            obj.docking_pose, pair)
        else:
            pass

    def _comparsion(self, aminoacid1, aminoacid2, obj):
        """
        """
        # print(
        #     rmsd_calc(self.coordinates[int(aminoacid1)],
        #               obj.coordinates[int(aminoacid2)]))
        return rmsd_calc(self.coordinates[int(aminoacid1)],
                         obj.coordinates[int(aminoacid2)])

    def N_d(self, aminoacid1, aminoacid2, obj):
        """
        """
        # print(
        #     rmsd_calc(self.coordinates[int(aminoacid1)][0],
        #               obj.coordinates[int(aminoacid2)][0]))
        return rmsd_calc(self.coordinates[int(aminoacid1)][0],
                         obj.coordinates[int(aminoacid2)][0])

    def vec_d(self, aminoacid1, aminoacid2, obj):
        """
        """
        vec1 = self.coordinates[int(aminoacid1)][0] - self.coordinates[int(
            aminoacid1)][3]
        vec2 = obj.coordinates[int(aminoacid1)][0] - obj.coordinates[int(
            aminoacid1)][3]
        # print(rmsd_calc(vec1, vec2))
        return rmsd_calc(vec1, vec2)

    def comparsion(self, obj):
        all_data = []
        all_pairs = pair_creator(self.peptides, obj.peptides)
        if all_pairs:
            if all_pairs[0]:
                if self.comparsion_one(pairs=all_pairs[0], obj=obj):
                    all_data.append(
                        tuple(
                            self.comparsion_one(pairs=all_pairs[0], obj=obj)))
            if all_pairs[1]:
                if self.comparsion_two(pairs=all_pairs[1], obj=obj):
                    all_data.append(
                        tuple(
                            self.comparsion_two(pairs=all_pairs[1], obj=obj)))
        return tuple(all_data)


def peptides_init(path_to_hdf5: str):
    """
    Функция для инициализации пептидов.
    В файле 'path_to_hdf5' должны храниться
    данные о N Ca C O атомах аминокислот пептидов, а так же
    "Имя" пептида (в качестве ключа) и номер его докинг позы
    (AAA/1/np.random.rand(12,3))
    """
    with h5py.File(path_to_hdf5, 'r') as data_file:
        all_data_from_hdf5 = []
        all_keys = data_file.keys()
        for key in tqdm(all_keys):
            one_peptide_data = [
                Tripep(
                    name=key,
                    conf=str(x),
                    coordinates=data_file[key][str(x)][:]) for x in range(10)
            ]
            all_data_from_hdf5.append(one_peptide_data)
        return tuple(all_data_from_hdf5)


def peptides_process(peptides_list: tuple, new_file_name: str):
    '''
    Функция для обработки инициализированных пептидов
    '''
    with open(new_file_name, 'w') as new_file:
        for pep1_indx in tqdm(range(len(peptides_list))):
            pep1 = peptides_list[pep1_indx][0]
            all_peps_for_pep1 = walker(pep1.peptides)

            for pep2_indx in range(len((peptides_list))):
                pep2 = peptides_list[pep2_indx][0]

                if set(pep1.peptides) & set(pep2.peptides):
                    pass
                else:
                    continue
                if pep2.peptides not in all_peps_for_pep1:
                    # if not pair_creator(pep1.peptides, pep2.peptides):
                    continue
                for conf1 in peptides_list[pep1_indx]:
                    for conf2 in peptides_list[pep2_indx]:
                        comp_result = conf1.comparsion_uni(conf2)
                        if comp_result:
                            new_file.write(str(comp_result) + '\n')


def peptides_process_mpi(peptides_list: tuple, new_file_name: str):
    '''
    Функция для обработки инициализированных пептидов
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    pep1_indx = rank
    with open(str(rank) + new_file_name, 'a+') as new_file:
        if rank == 0:
            pbar = tqdm(total=len(peptides_list))
        while pep1_indx < len(peptides_list):
            if rank == 0:
                pbar.update(pep1_indx)
            pep1 = peptides_list[pep1_indx][0]
            all_peps_for_pep1 = walker(pep1.peptides)
            for pep2_indx in range(len((peptides_list))):
                pep2 = peptides_list[pep2_indx][0]
                if set(pep1.peptides) & set(pep2.peptides):
                    pass
                else:
                    continue
                if pep2.peptides not in all_peps_for_pep1:
                    # if not pair_creator(pep1.peptides, pep2.peptides):
                    continue
                for conf1 in peptides_list[pep1_indx]:
                    for conf2 in peptides_list[pep2_indx]:
                        comp_result = conf1.comparsion_uni(conf2)
                        if comp_result:
                            new_file.write(str(comp_result) + '\n')
            pep1_indx += size
        if rank == 0:
            pbar.close()


def ag_pars():
    parser = ag.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input-file',
        required=True,
        dest='path_to_hdf5',
        type=str,
        help='Input file with peptides')

    parser.add_argument(
        '-o',
        '--output-file',
        required=True,
        dest='new_file_name',
        type=str,
        help='Relative path to new file(s)')

    args = vars(parser.parse_args())
    return args


def main():
    args = ag_pars()
    data = peptides_init(args['path_to_hdf5'])
    peptides_process_mpi(
        peptides_list=data, new_file_name=args['new_file_name'])


if __name__ == '__main__':
    main()
