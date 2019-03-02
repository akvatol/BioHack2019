#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sh


class Parser:
    __slots__ = ('all_edges', 'all_nodes', 'file')

    def __init__(self, path: str):
        self.file = path
        self.all_nodes = []
        self.all_edges = []

    def line_parser(self, string: str):
        new_str = new_str = string.split(',')[:5]
        node1 = new_str[0][1:-1] + new_str[1][2:-1]
        node2 = new_str[2][2:-1] + new_str[3][2:-1]
        if node1 in self.all_nodes:
            pass
        else:
            self.all_nodes.append(node1)
        if node2 in self.all_nodes:
            pass
        else:
            self.all_nodes.append(node2)

        pair = new_str[4][2:-1]

        if int(pair[0:2]) > int(pair[2:]):
            print(len(pair) // 2)
            print(pair[0:2], pair[2:])

        # if int(new_str[4][0:len(new_str[4][2:-1]) / 2]) > int(
        #         new_str[4][len(new_str[4][2:-1]) / 2:]):
        #     edge = (self.all_nodes.index(node1), self.all_nodes.index(node2))
        # else:
        #     edge = (self.all_nodes.index(node2), self.all_nodes.index(node1))
        # self.all_edges.append(edge)


p = Parser(path='sttt')
p.line_parser(
    "'AAA', '0', 'AAC', '4', '1201', 1.0911943620554276, 0.9602496815368986'")
print(p.all_nodes, p.all_edges)

# def parse_line(string: str):
#     # 'AAA', '0', 'AAC', '4', '1201', 1.0911943620554276, 0.9602496815368986
#     # Возвращает 2 ноды и связь между ними
#     new_str = string.split(',')[:5]
#     node1 = new_str[0][1:-1] + new_str[1][1:-1]
#     node2 = new_str[2][1:-1] + new_str[3][1:-1]
#     if int(new_str[4][0:len(new_str[4])/2]) > int(new_str[4][len(new_str[4])/2:]):
#          edge = (node1, node2)
#     else:
#         edge = (node2, node1)
#
#
#
# def parse_file(path: str):
#     pass
#
#
# #%%
#


def time(func):
    import time

    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        res = func(*args, **kwargs)
        print(func.__name__, time.perf_counter() - t)
        return res

    return wrapper


@time
def some():
    with open('src/2peptides_clear.csv', 'r') as f:
        print(sum(1 for _ in f))


some()


#%%
