# coding: utf-8
from itertools import combinations
from simpleai.search import SearchProblem
from simpleai.search.local import hill_climbing
# from pgm.my_bayesnet.scores import bic_score


class LocalSearch(SearchProblem):

    def __init__(self, data, *args, **kwargs):
        """
        seules les données sont nécessaires, afin de pouvoir scorer
        en respectant l'interface
        """
        super(LocalSearch, self).__init__(*args, **kwargs)
        self.data = data

    def actions(self, state):
        """
        les actions possibles sont l'ajout, la suppression ou le retournement
        d'un arc. La vérification que l'ajout ou le retournement d'arcs ne
        crée pas de cycle n'est pas fait ici, mais dans le result.

        une action est un triplet de variables (ix_a, ix_b, act), avec
        act = "add", "del", "rev" indiquant l'un des 3 moficications
        possible pour l'arc (a,b)
        """
        acts = []
        for i, j in combinations(range(state.shape[0]), 2):
            if state[i, j] == 0:
                acts.append((i, j, "add"))
                acts.append((j, i, "add"))
            else:
                acts.append((i, j, "del"))
                acts.append((i, j, "rev"))
        return acts

    def result(self, state, action):
        """
        retourne un nouvel état en appliquant l'action considérée
        """
        ni, nj, val = action
        try:
            if val == "add":
                return state.add_edge(ni, nj, on_copy=True)
            elif val == "del":
                return state.delete_edge(ni, nj, on_copy=True)
            else:
                return state.reverse_edge(ni, nj, on_copy=True)
        except AttributeError:
            return None

    def value(self, state):
        """
        Renvoie le score bic de la solution courante.
        Un socre élevé est meilleur.
        Si la solution a un cycle, on renvoie -infini. Ca ne risque pas de
        fausser l'algorithme (par exemple, si tous les candidats on un cycle),
        puisque la solution actuelle est correcte et ne présente pas de cycle.
        """
        import numpy as np
        return np.random.rand()
        # if state is None:
        #     return -float("inf")
        # if state.is_acyclic():
        #     return bic_score(state, self.data)
        # return -float("inf")

    def solve_hillclimbing(self, iterations_limit=0, viewer=None):
        """
        résolution du problème
        """
        return hill_climbing(self, iterations_limit, viewer)








import numpy as np

class AdjacencyBayesStruct(np.ndarray):

    def __new__(cls, a, dtype=int, n_values=None):
        obj = np.asarray(a, dtype).view(cls)
        obj.n_values = n_values
        obj.topological_order = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.n_values = getattr(obj, 'n_values', None)
        if self.n_values is None:
            self.n_values = 2 * np.ones(self.shape[0])
        self.topological_order = None

    def __copy__(self):
        return AdjacencyBayesStruct(self.copy(), dtype=self.dtype,
                                    n_values=self.n_values)

    def add_node(self, n_values):
        """
        <private> ajout d'un noeud à la structure
        """
        copy = self.__copy__()
        copy.reshape((self.shape[0] + 1, self.shape[1] + 1))
        copy[-1, :] = 0
        copy[:, -1] = 0
        copy.n_values = np.r_[self.n_values, n_values]
        return copy

    def remove_node(self, ix_node):
        """
        supprime un noeud, retourne une copie
        """
        copy = np.delete(self, ix_node, 0)
        copy = np.delete(copy, ix_node, 1)
        new_values = np.delete(self.values, ix_node)
        return AdjacencyBayesStruct(copy, n_values=new_values)

    def add_edge(self, ix_child, ix_parent, on_copy=True):
        """
        ajoute un arc entre child et parent
        """
        if ix_parent == ix_child:
            raise ValueError("add edge with parent=child")
        if on_copy:
            copy = self.__copy__()
            copy[ix_parent, ix_child] = 1
            copy[ix_child, ix_parent] = -1
            copy._build_topo_order()
            if not copy.is_acyclic():
                raise CyclicGraphException()
            return copy
        self[ix_parent, ix_child] = 1
        self[ix_child, ix_parent] = -1
        self._build_topo_order()
        if not self.is_acyclic():
            self[ix_parent, ix_child] = 0
            self[ix_child, ix_parent] = 0
            self._build_topo_order()
            raise CyclicGraphException()
        return self

    def remove_edge(self, ix_child, ix_parent, on_copy=True):
        """
        enlève un arc entre child et parent
        """
        if ix_parent == ix_child:
            raise ValueError("remove edge with parent=child")
        if on_copy:
            copy = self.__copy__()
            copy[ix_child, ix_parent] = 0
            copy[ix_parent, ix_child] = 0
            return copy
        self[ix_child, ix_parent] = 0
        self[ix_parent, ix_child] = 0
        return self

    def reverse_edge(self, ix_child, ix_parent, on_copy=True):
        """
        renverse un arc entre
        """
        if ix_parent == ix_child:
            raise ValueError("reverse edge with parent=child")
        if on_copy:
            copy = self.__copy__()
            copy[ix_child, ix_parent] = 1
            copy[ix_parent, ix_child] = -1
            copy._build_topo_order()
            if not copy.is_acyclic():
                raise CyclicGraphException()
            return copy
        self[ix_child, ix_parent] = 1
        self[ix_parent, ix_child] = -1
        self._build_topo_order()
        if not self.is_acyclic():
            self[ix_child, ix_parent] = 1
            self[ix_parent, ix_child] = -1
            self._build_topo_order()
            raise CyclicGraphException()
        return self

    def is_acyclic(self):
        """
        check if self est acyclic ou pas,
        il y a un cycle s'il n'est pas possible de trouver un order
        """
        if self.topological_order is None:
            self._build_topo_order()
        return len(self.topological_order) > 0

    def _build_topo_order(self):
        """
        Tentative de construction d'un ordonancement topologique
        """
        topo = []
        isnotchilds = [ix_node for ix_node in xrange(self.shape[0])
                      if -1 not in self[ix_node, :]]
        copy = self.__copy__()
        while isnotchilds:
            topo.append(isnotchilds.pop())
            for m in [ix for ix in xrange(copy.shape[0])
                      if copy[topo[-1], ix] == 1]:
                copy.remove_edge(m, topo[-1], False)
                if -1 not in copy[m, :]:
                    isnotchilds.append(m)
        if np.sum(np.abs(copy)) == 0:
            self.topological_order = np.array(topo)
        else:
            self.topological_order = []



class CyclicGraphException(Exception):
    pass








import pandas as pd
import numpy as np

# génération de data, façon lourdingue
size = 1000
df = pd.DataFrame({"Difficulty": np.random.choice([0, 1], size=size,
                                                  p=[.6, .4]),
                   "Intelligence": np.random.choice([0, 1], size=size,
                                                    p=[.7, .3])})
sat = {0: [.95, .05], 1: [.2, .8]}
df["SAT"] = df.apply(lambda ctx: np.random.choice(
    [0, 1], p=sat[ctx["Intelligence"]]), axis=1)
grade = {(0, 0): [.3, .4, .3], (1, 0): [.05, .25, .7],
         (0, 1): [.9, .08, .02], (1, 1): [.5, .3, .2]}
df["grade"] = df.apply(lambda ctx: np.random.choice(
    [0, 1, 2], p=grade[(ctx["Difficulty"], ctx["Intelligence"])]), axis=1)
letter = {0: [.1, .9], 1: [.4, .6], 2: [.99, .01]}
df["letter"] = df.apply(
    lambda ctx: np.random.choice([0, 1], p=letter[ctx["grade"]]), axis=1)


bn0 = AdjacencyBayesStruct(np.zeros((5, 5)), n_values=[2, 2, 2, 3, 2])
problem = LocalSearch(initial_state=bn0, data=df)
result = problem.solve_hillclimbing(5)


