HEURISTIC_MAP = {
    'Oradea': 380,
    'Zerind': 374,
    'Arad': 366,
    'Timisoara': 329,
    'Lugoj': 244,
    'Mehadia': 241,
    'Drobeta': 242,
    'Craiova': 160,
    'Rimnicu_Vilcea': 193,
    'Pitesti': 100,
    'Sibiu': 253,
    'Fagaras': 176,
    'Giurgiu': 77,
    'Urziceni': 80,
    'Hirsova': 151,
    'Eforie': 161,
    'Vaslui': 199,
    'Iasi': 226,
    'Neamt': 234,
    'Bucharest': 0
}

GRAPH_TRANSITIONS = {
    'Arad': [('Zerind', 75), ('Timisoara', 118), ('Sibiu', 140)],
    'Bucharest': [('Giurgiu', 90), ('Urziceni', 85), ('Fagaras', 211), ('Pitesti', 101)],
    'Craiova': [('Drobeta', 120), ('Rimnicu_Vilcea', 146), ('Pitesti', 138), ('Drobeta', 120)],
    'Drobeta': [('Mehadia', 75), ('Craiova', 120)],
    'Eforie': [('Hirsova', 86)],
    'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
    'Hirsova': [('Urziceni', 98), ('Eforie', 86)],
    'Iasi': [('Neamt', 87), ('Vaslui', 92)],
    'Lugoj': [('Timisoara', 111), ('Mehadia', 70)],
    'Oradea': [('Zerind', 71), ('Sibiu', 151)],
    'Pitesti': [('Rimnicu_Vilcea', 97), ('Bucharest', 101), ('Craiova', 138)],
    'Rimnicu_Vilcea': [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)],
    'Urziceni': [('Vaslui', 142), ('Bucharest', 85), ('Hirsova', 98)],
    'Zerind': [('Arad', 75), ('Oradea', 71)],
    'Timisoara': [('Arad', 118), ('Lugoj', 111)],
    'Sibiu': [('Arad', 140), ('Fagaras', 99), ('Oradea', 151), ('Rimnicu_Vilcea', 80)],
    'Giurgiu': [('Bucharest', 90)],
    'Mehadia': [('Drobeta', 75), ('Lugoj', 70)],
    'Neamt': [('Iasi', 87)],
    'Vaslui': [('Iasi', 92), ('Urziceni', 142)]
}

OPTIMAL_PATH_FROM_ARAD = [
    'Arad',
    'Sibiu',
    'Rimnicu_Vilcea',
    'Pitesti',
    'Bucharest'
]

OPTIMAL_COSTS_FROM_ARAD = {
    'Arad': 0,
    'Zerind': 75,
    'Timisoara': 118,
    'Sibiu': 140,
    'Oradea': 146,
    'Fagaras': 239,
    'Rimnicu_Vilcea': 220,
    'Craiova': 366,
    'Pitesti': 317,
    'Bucharest': 418,
    'Lugoj': 229,
    'Mehadia': 299,
    'Drobeta': 374,
    'Giurgiu': 508,
    'Urziceni': 503,
    'Hirsova': 601,
    'Eforie': 687,
    'Vaslui': 645,
    'Iasi': 737,
    'Neamt': 824
}
