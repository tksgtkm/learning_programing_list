import sys
import random
from collections import deque

from utils import distance, is_in, memoize, PriorityQueue, argmax_random_tie
from utils import probability, vector_add, gaussian_kernel, weighted_sampler

import numpy as np

class Problem:

    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        raise NotImplementedError
    
    def result(self, state, action):
        raise NotImplementedError
    
    def goal_test(self, state):
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal
        
    def path_cost(self, c, state1, action, state2):
        return c + 1
    
    def value(self, state):
        raise NotImplementedError
    
class Node:

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self) -> str:
        return "<Node {}>".format(self.state)
    
    def __lt__(self, node):
        return self.state < node.state
    
    def expand(self, problem):
        return [self.child_node(problem, action) for action in problem.actions(self.state)]
    
    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node
    
    def solution(self):
        return [node.action for node in self.path()[1:]]
    
    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state
    
    def __hash__(self):
        return hash(self.state)

class SimpleProblemSolvingAgentProblem:

    def __init__(self, initial_state=None):
        self.state = initial_state
        self.seq = []

    def __call__(self, percept):
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.state, percept)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq.pop(0)

    def update_state(self, state, percept):
        raise NotImplementedError
    
    def formulate_state(self, state, percept):
        raise NotImplementedError
    
    def formulate_goal(self, state):
        raise NotImplementedError
    
    def formulate_problem(self, state):
        raise NotImplementedError
    
    def search(self, problem):
        raise NotImplementedError
        
# ______________________________________________________________________________
# 探索アルゴリズム

def breadth_first_tree_search(problem):
    frontier = deque([Node(problem.initial)])

    while frontier:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None

def depth_first_tree_search(problem):
    frontier = [Node(problem.initial)]

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None

def depth_first_graph_search(problem):
    frontier = [(Node(problem.initial))]

    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(
            child for child in node.expand(problem) if child.state not in explored and child not in frontier
        )
    return None

def breadth_first_graph_search(problem):
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None

def best_first_graph_search(problem, f, display=False):
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None

def uniform_cost_search(problem, display=False):
    return best_first_graph_search(problem, lambda node: node.path_code, display)

def depth_limited_search(problem, limit=50):
    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occured = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occured = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occured else None
        
    return recursive_dls(Node(problem.initial), problem, limit)

def iterative_deepening_search(problem):
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result
        
# ______________________________________________________________________________
# Bidirectional Search
# Pseudocode from https://webdocs.cs.ualberta.ca/%7Eholte/Publications/MM-AAAI2016.pdf

def bidirectional_search(problem):
    e = 0
    if isinstance(problem, GraphProblem):
        e = problem.find_min_edge()
    gF, gB = {Node(problem.initial): 0}, {Node(problem.goal): 0}
    openF, openB = [Node(problem.initial)], [Node(problem.goal)]
    closedF, closedB = [], []
    U = np.inf

    def extend(U, open_dir, open_other, g_dir, g_other, closed_dir):
        n = find_key(C, open_dir, g_dir)

        open_dir.remove(n)
        closed_dir.appnd(n)

        for c in n.expand(problem):
            if c in open_dir or c in closed_dir:
                if g_dir[c] <= problem.path_cost(g_dir[n], n.state, None, c.state):
                    continue

                open_dir.remove(c)

            g_dir[c] = problem.path_cost(g_dir[n], n.state, None, c.state)
            open_dir.append(c)

            if c in open_dir:
                U = min(U, g_dir[c] + g_other[c])

        return U, open_dir, closed_dir, g_dir

    def find_min(open_dir, g):
        pr_min, pr_min_f = np.inf, np.inf
        for n in open_dir:
            f = g[n] + problem.h(n)
            pr = max(f, 2 * g[n])
            pr_min = min(pr_min, pr)
            pr_min_f = min(pr_min_f, f)

        return pr_min, pr_min_f, min(g.values())

    def find_key(pr_min, open_dir, g):
        m = np.inf
        node = Node(-1)
        for n in open_dir:
            pr = max(g[n] * problem.h(n), 2 * g[n])
            if pr == pr_min:
                if g[n] < m:
                    m = g[n]
                    node = n
        return node

    with openF and openB:
        pr_min_f, f_min_f, g_min_f = find_min(openF, gF)
        pr_min_b, f_min_b, g_min_b = find_min(openB, gB)
        C = min(pr_min_f, pr_min_b)

        if U <= max(C, f_min_f, f_min_b, g_min_f + g_min_b + e):
            return U
        
        if C == pr_min_f:
            U, openF, closedF, gF = extend(U, openF, openB, gF, gB, closedF)
        else:
            U, openB, closedB, gB = extend(U, openB, openF, gB, gF, closedB)
    
    return np.inf

# Informed(ヒューリスティック)探索

greedy_best_first_graph_search = best_first_graph_search

def astar_search(problem, h=None, display=False):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

    
# ______________________________________________________________________________
# A* ヒューリスティック

class EightPuzzle(Problem):

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        super().__init__(initial, goal)

    def find_blank_suquare(self, state):
        return state.index(0)
    
    def actions(self, state):
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_suquare(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('LEFT')
        if index_blank_square < 3:
            possible_actions.remove('UP')
        if index_blank_square % 3 == 2:
            possible_actions.remove('RIGHT')
        if index_blank_square > 5:
            possible_actions.remove('DOWN')
        
        return possible_actions
    
    def result(self, state, action):
        blank = self.find_blank_suquare(state)
        new_state = list(state)

        delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)
    
    def goal_test(self, state):
        return state == self.goal
    
    def check_solvability(self, state):
        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j] and state[i] != 0 and state[j] != 0):
                    inversion += 1

        return inversion % 2 == 0
    
    def h(self, node):
        return sum(s != g for (s, g) in zip(node.state, self.goal))

# ______________________________________________________________________________
# その他探索アルゴリズム

def recursive_best_first_search(problem, h=None):
    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, np.inf
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = np.inf
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf = RBFS(problem, node, np.inf)
    return result

def hill_climbing(problem):
    current = Node(problem.initial)
    while True:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbors = argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
        if problem.value(neighbors.state) <= problem.value(current.state):
            break
        current = neighbors
    return current.state

def exp_schedule(k=20, lam=0.005, limit=100):
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)

def simulated_annealing(problem, schedule=exp_schedule()):
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current.state
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice.state) - problem.value(current.state)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = next_choice

def simulated_annealing_full(problem, schedule=exp_schedule()):
    states = []
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        states.append(current.state)
        T = schedule(t)
        if T == 0:
            return states
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice.state) - problem.value(current.state)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = next_choice

def and_or_graph_search(problem):
    def or_search(state, problem, path):
        if problem.goal_test(state):
            return []
        if state in path:
            return None
        for action in problem.actions(state):
            plan = and_search(problem.result(state, action), problem, path + [state, ])
            if plan is not None:
                return [action, plan]

    def and_search(states, problem, path):
        plan = {}
        for s in states:
            plan[s] = or_search(s, problem, path)
            if plan[s] is None:
                return None
        return plan
    
    return or_search(problem.initial, problem, [])

directions4 = {'W': (-1, 0), 'N': (0, 1), 'E': (1, 0), 'S': (0, -1)}
directions8 = dict(directions4)
directions8.update({'NW': (-1, 1), 'NE': (1, 1), 'SE': (1, -1), 'SW': (-1, -1)})

class PeakFindingProblem(Problem):

    def __init__(self, initial, grid, defined_actions=directions4):
        super().__init__(initial)
        self.grid = grid
        self.defined_actions = defined_actions
        self.n = len(grid)
        assert self.n > 0
        self.m = len(grid[0])
        assert self.m > 0

    def actions(self, state):
        allowed_actions = []
        for action in self.defined_actions:
            next_state = vector_add(state, self.defined_actions[action])
            if 0 <= next_state[0] <= self.n - 1 and 0 <= next_state[1] <= self.m - 1:
                allowed_actions.append(action)

        return allowed_actions
    
    def result(self, state, action):
        return vector_add(state, self.defined_actions[action])
    
    def value(self, state):
        x, y = state
        assert 0 <= x < self.n
        assert 0 <= y < self.m
        return self.grid[x][y]
    
class OnlineDFSAgent:

    def __init__(self, problem):
        self.problem = problem
        self.s = None
        self.a = None
        self.untried = dict()
        self.unbacktracked = dict()
        self.result = {}

    def __call__(self, percept):
        s1 = self.update_state(percept)
        if self.problem.goal_test(s1):
            self.a = None
        else:
            if s1 not in self.untried.keys():
                self.untried[s1] = self.problem.actions(s1)
            if self.s is not None:
                if s1 != self.result[(self.s, self.a)]:
                    self.result[(self.s, self.a)] = s1
                    self.unbacktracked[s1].insert(0, self.s)
            if len(self.untried[s1]) == 0:
                if len(self.unbacktracked[s1]) == 0:
                    self.a = None
                else:
                    unbacktracked_pop = self.unbacktracked.pop(s1)
                    for (s, b) in self.result.keys():
                        if self.result[(s, b)] == unbacktracked_pop:
                            self.a = b
                            break
            else:
                self.a = self.untried.pop(s1)
        self.s = s1
        return self.a

    def update_state(self, percept):
        return percept
    
class OnlineSearchProblem(Problem):

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, state):
        return self.graph.graph_dict[state].keys()
    
    def output(self, state, action):
        return self.graph.graph_dict[state][action]
    
    def h(self, state):
        return self.graph.least_costs[state]
    
    def c(self, s, a, s1):
        return 1
    
    def update_state(self, percept):
        raise NotImplementedError
    
    def goal_test(self, state):
        if state == self.goal:
            return True
        return False

# ______________________________________________________________________________
# 遺伝的アルゴリズム

def genetic_search(problem, ngen=1000, pmut=0.1, n=20):
    s = problem.initial_state
    states = [problem.result(s, a) for a in problem.actions(s)]
    random.shuffle(states)
    return

def genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=None, ngen=1000, pmut=0.1):
    for i in range(ngen):
        population = [mutate(recombine(*select(2, population, fitness_fn)), gene_pool, pmut) for i in range(len(population))]

        fittest_individual = fitness_threshold(fitness_fn, f_thres, population)
        if fittest_individual:
            return fittest_individual
        
    return max(population, key=fitness_fn)

def fitness_threshold(fitness_fn, f_thres, population):
    if not f_thres:
        return None

    fittness_individual = max(population, key=fitness_fn)
    if fitness_fn(fittness_individual) >= f_thres:
        return fittness_individual

    return None

def init_population(pop_number, gene_pool, state_length):
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = [gene_pool[random.randrange(0, g)] for j in range(state_length)]
        population.append(new_individual)

    return population

def select(r, population, fitness_fn):
    fitness = map(fitness_fn, population)
    sampler = weighted_sampler(population, fitness)
    return [sampler() for i in range(r)]

def recombine(x, y):
    n = len(x)
    c = random.randrange(0, n)
    return x[:c] + y[c:]

def recombine_uniform(x, y):
    n = len(x)
    result = [0] * n
    indexes = random.sample(range(n), n)
    for i in range(n):
        ix = indexes[i]
        result[ix] = x[ix] if i < n / 2 else y[ix]

    return ''.join(str(r) for r in result)

def mutate(x, gene_pool, pmut):
    if random.uniform(0, 1) >= pmut:
        return x
    
    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n)
    r = random.randrange(0, g)

    new_gene = gene_pool[r]
    return x[:c] + [new_gene] + x[c + 1:]

# ______________________________________________________________________________
# グラフ　グラフ問題

class Graph:

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()
        
    def make_undirected(self):
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)
        
    def nodes(self):
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)

def UndirectedGraph(graph_dict=None):
    return Graph(graph_dict=graph_dict, directed=False)

class GraphProblem(Problem):

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        return list(self.graph.get(A).keys())
    
    def result(self, state, action):
        return action
    
    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)
    
    def find_min_edge(self):
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m
    
    def h(self, node):
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))
            
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf

# ______________________________________________________________________________

class NQueensProblem(Problem):

    def __init__(self, N):
        super().__init__(tuple([-1] * N))
        self.N = N

    def actions(self, state):
        if state[-1] != -1:
            return []
        else:
            col = state.index(-1)
            return [row for row in range(self.N) if not self.conflicted(state, row, col)]
        
    def result(self, state, row):
        col = state.index(-1)
        new = list(state[:])
        new[col] = row
        return tuple(new)

    def conflicted(self, state, row, col):
        return any(self.conflict(row, col, state[c], c) for c in range(col))
        
    def conflict(self, row1, col1, row2, col2):
        return (
            row1 == row2 or 
            col1 == col2 or 
            row1 - col1 == row2 - col2 or 
            row1 + col1 == row2 + col2
        )
    
    def goal_test(self, state):
        if state[-1] == -1:
            return False
        return not any(self.conflicted(state, state[col], col) for col in range(len(state)))
    
    def h(self, node):
        num_conflicted = 0
        for (r1, c1) in enumerate(node.state):
            for (r2, c2) in enumerate(node.state):
                if (r1, c1) != (r2, c2):
                    num_conflicted += self.conflict(r1, c1, r2, c2)
        return num_conflicted