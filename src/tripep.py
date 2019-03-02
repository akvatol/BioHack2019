# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from itertools import product
import h5py
from tqdm import tqdm
from mpi4py import MPI

TRASHHOLD = 1

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
    print('DEBUG: min_intersec= ', min_intersec)
    for i in range(1, min_intersec):
        if second_pep.endswith(first_pep[:i]):
            all_pairs.append(''.join([str(x) for x in range(i)]) + ''.join(
                [str(x) for x in reversed(range(len(second_pep) - i, len(second_pep)))]))
            print(first_pep[:i])
            print('DEBUG: = ', '1')
        if first_pep.endswith(second_pep[:i]):
            #all_pairs.append()
            print(second_pep[:i])
            print('DEBUG: = ', '2')

    return tuple(all_pairs)


print(pair_creator('CAA', 'AAA'))
# %%


'''
    if first_pep[0] == second_pep[-1]
    if first_pep[0] == second_pep[-1]:
        pair = str(0) + str(len(second_pep) - 1)
        one_aa.append(pair)
    if first_pep[-1] == second_pep[0]:
        pair = str(len(second_pep) - 1) + str(0)
        one_aa.append(pair)
    if first_pep[0:2] == second_pep[-2:]:
        pair2 = str('01') + str(len(second_pep) - 2) + str(len(second_pep) - 1)
        two_aa.append(pair2)
    if first_pep[-2:] == second_pep[0:2]:
        pair2 = str(len(second_pep) - 2) + str(len(second_pep) - 1) + str('01')
        two_aa.append(pair2)
    if one_aa or two_aa:
        return (tuple(one_aa), tuple(two_aa))
'''


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
            Матрица 12 х 3 с координатами (N Ca C O)
        """
        self.peptides = name
        self.docking_pose = conf
        self.coordinates = chunkit(data=coordinates, n=len(name))

    def comparsion_one(self, pairs: tuple, obj=None):
        """
        pairs: tuple
            Пары одинаковых аминокислот пиптида, для
            которых нужно посчитать RMSD, первое число --
            номер аминокислоты этого пептида, вторая --
            номер аминокислоты пиптида с которым сравниваем.
        """

        list_of_good_pairs = []
        for pair in pairs:
            rms = rmsd_calc(self.coordinates[int(pair[0])],
                            obj.coordinates[int(pair[1])])
            if rms <= TRASHHOLD:
                list_of_good_pairs.append(
                    (self.peptides, self.docking_pose, obj.peptides,
                     obj.docking_pose, pair, rms))
        return (list_of_good_pairs)

    def comparsion_two(self, pairs: tuple, obj):
        """
        Функция для расчета RMSD для ПАР элементов
        (Пептидов у которых пересечение по ДВУМ аминокислотам)
        """
        # Переделать, чтобы после первого нахождения прерывалось
        list_of_good_pairs = []
        for pair in pairs:
            rms1 = rmsd_calc(self.coordinates[int(pair[0])],
                             obj.coordinates[int(pair[2])])
            rms2 = rmsd_calc(self.coordinates[int(pair[1])],
                             obj.coordinates[int(pair[3])])
            if rms1 <= TRASHHOLD and rms2 <= TRASHHOLD:
                list_of_good_pairs.append(
                    (self.peptides, self.docking_pose, obj.peptides,
                     obj.docking_pose, pair, rms1, rms2))
        return (list_of_good_pairs)

    def comparsion_three(self, pairs: tuple, obj):

        list_of_good_pairs = []
        for pair in pairs:
            rms1 = rmsd_calc(self.coordinates[int(pair[0])],
                             obj.coordinates[int(pair[2])])
            rms2 = rmsd_calc(self.coordinates[int(pair[1])],
                             obj.coordinates[int(pair[3])])
            if rms1 <= TRASHHOLD and rms2 <= TRASHHOLD:
                list_of_good_pairs.append(
                    (self.peptides, self.docking_pose, obj.peptides,
                     obj.docking_pose, pair, rms1, rms2))
        return (list_of_good_pairs)

    def comparsion_uni(self, obj): pass

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
    with h5py.File(path_to_hdf5, 'r') as data_file:
        all_data_from_hdf5 = []
        for key in tqdm(data_file.keys()):
            one_peptide_data = [
                Tripep(
                    name=key,
                    conf=str(x),
                    coordinates=data_file[key][str(x)][:]) for x in range(10)
            ]
            all_data_from_hdf5.append(one_peptide_data)
        return tuple(all_data_from_hdf5)


def peptides_process(peptides_list: tuple, new_file_name: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = []
    with open(new_file_name, 'w') as file:
        pep1 = rank
        while pep1 < len(peptides_list):
            print(str(rank))
        # for pep1 in tqdm(range(len(peptides_list))):
            all_peps_for_pep1 = walker(peptides_list[pep1][0].peptides)
            for pep2 in rangelen((peptides_list)):
                if peptides_list[pep2][0].peptides in all_peps_for_pep1:
                    for x in peptides_list[pep1]:
                        for y in peptides_list[pep2]:
                            c = x.comparsion(y)
                            if c:
                                data.append(str(c))
            pep1 += size

    if rank > 0:
        comm.send(data)
    elif rank == 0:
        for i in range(1, size):
            tmpdata = comm.recv(source=i)
            data.extend(tmpdata)

        file.write("\n".join(data))


def main():
    data = peptides_init('/home/antond/projects/BioHack2019/data/12x3.hdf5')
    peptides_process(
        peptides_list=data,
        new_file_name='/home/antond/projects/BioHack2019/src/test_mpi.txt')


if __name__ == '__main__':
    # main()

    # %%
