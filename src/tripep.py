#%%
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from numba import jit
from itertools import product

TRASHHOLD = 1.5

amino_alphabet = [
    'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'S', 'T', 'N', 'Q', 'C', 'G', 'P',
    'R', 'H', 'K', 'D', 'E'
]


def get_dipep(k=2):
    dipep = [''.join(kmer) for kmer in product(amino_alphabet, repeat=k)]
    return dipep


def walker(pep_type: str):
    pep_list = []
    for i in range(0, 20):
        pos_pep = pep_type[0:2] + amino_alphabet[i]
        pep_list.append(pos_pep)
        i += 1
    for j in range(0, len(get_dipep())):
        pos_pep = pep_type[0] + get_dipep()[j]
        pep_list.append(pos_pep)
        j += 1
    cutted_pep_list = set(pep_list)
    cutted_pep_list.remove(pep_type)
    return cutted_pep_list


def pair_creator(first_pep: str, second_pep: str):
    """
    Принимает на вход два пептида и возвращает все возможные пары 
    комбинаций
    """
    one_aa = []
    two_aa = []
    if first_pep[0] == second_pep[2]:
        pair = str(0) + str(2)
        one_aa.append(pair)
    if first_pep[2] == second_pep[0]:
        pair = str(2) + str(0)
        one_aa.append(pair)
    if first_pep[0:2] == second_pep[1:3]:
        pair2 = str('01') + str('12')
        two_aa.append(pair2)
    if first_pep[1:3] == second_pep[0:2]:
        pair2 = str('12') + str('01')
        two_aa.append(pair2)
    if one_aa or two_aa:
        return (tuple(one_aa), tuple(two_aa))


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


def time(func):
    import time

    def wrapper(*args, **kwargs):
        t = time.clock()
        res = func(*args, **kwargs)
        print(func.__name__, time.clock() - t)
        return res

    return wrapper


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
                list_of_good_pairs.append([
                    self.peptides, self.docking_pose, obj.peptides,
                    obj.docking_pose, pair, rms
                ])
        return list_of_good_pairs

    def comparsion_two(self, pairs: tuple, obj):
        """
        Функция для расчета RMSD для ПАР элементов
        (Пептидов у которых пересечение по ДВУМ аминокислотам) 
        """
        # Переделать, чтобы после первого нахождения прерывалось
        list_of_good_pairs = []
        for pair in pairs:
            rms1 = None
            rms2 = None
            if rms1 <= TRASHHOLD and rms2 <= TRASHHOLD:
                list_of_good_pairs.append([
                    self.peptides, self.docking_pose, obj.peptides,
                    obj.docking_pose, pair, rms1, rms2
                ])
        return list_of_good_pairs
    
    def comparsion(self, obj):
        all_data = []
        all_pairs = pair_creator(self.peptides, obj.peptides)
        if all_pairs:
            if all_pairs[0]:
                all_data.append(self.comparsion_one(pairs=all_pairs[0], obj=obj))
            if all_pairs[1]:
                all_data.append(self.comparsion_two(pairs=all_pairs[1], obj=obj))
        return all_data


#%%

a = Tripep(name='AAA', conf='1', coordinates=np.random.rand(12, 3))
b = Tripep(name='ABA', conf='2', coordinates=np.random.rand(12, 3))

a.comparsion(b)

#%%
