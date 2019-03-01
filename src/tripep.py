#%%
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from numba import jit

TRASHHOLD = 1.5


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
        self.coordinates = chunkit(data = coordinates, n=len(name))

    def comparsion_one(self, pairs: tuple, obj = None):
        """
        pairs: tuple
            Пары одинаковых аминокислот пиптида, для
            которых нужно посчитать RMSD, первое число --
            номер аминокислоты этого пептида, вторая -- 
            номер аминокислоты пиптида с которым сравниваем.
        """
        list_of_good_pairs = []
        for pair in pairs:
            rms = rmsd_calc(self.coordinates[int(pair[0])], obj.coordinates[int(pair[1])])
            if rms <= TRASHHOLD:
                list_of_good_pairs.append([self.peptides,
                                           self.docking_pose,
                                           obj.peptides,
                                           obj.docking_pose,
                                           pair,
                                           rms])
        return list_of_good_pairs
            

    def comparsion_two(self, pairs: tuple, obj):
        """
        Функция для расчета RMSD для ПАР элементов
        (Пептидов у которых пересечение по ДВУМ аминокислотам) 
        """
        pass
        
        

    def comparsion_three(self, obj):
        pass

    @classmethod
    def from_dataset(cls, data=None):
        return cls

#%%

a = Tripep(name='AAA', conf='1', coordinates=np.random.rand(12,3))
b = Tripep(name='ABA', conf='2', coordinates=np.random.rand(12,3))

print(a.comparsion_one(pairs = ('00', '02', '20', '22'), obj=b))


#%%
