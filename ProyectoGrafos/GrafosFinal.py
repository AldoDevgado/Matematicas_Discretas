import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import heapq
from collections import deque
# ----------------------------
# Lectura de matriz de adyacencia
# ----------------------------
def leer_matriz(m):
    M = np.zeros((m, m), dtype=int)
    print(f"Ingrese la matriz de adyacencia {m}×{m}:")
    for i in range(m):
        while True:
            try:
                fila = list(map(int, input(f"Fila {i+1}: ").split()))
                if len(fila) != m or any(x not in (0,1) for x in fila):
                    raise ValueError
                M[i] = fila
                break
            except ValueError:
                print(" Error: ingresa exactamente m valores 0 o 1.")
    return M

# --- PROPIEDADES DEL GRAFO DIRIGIDO ---
def es_reflexiva(matriz):
    n = len(matriz)
    for i in range(n):
        if matriz[i][i] == 1:
            return True
    return False


def es_simetrica(matriz):
    n = len(matriz)  # Número de nodos
    for i in range(n):
        for j in range(n):
            if i != j and matriz[i][j] == 1 and matriz[j][i] == 1:
                return True
    return False


def es_antisimetrica(matriz):
   if es_simetrica(matriz):
       return False
   else:
       return True


def es_transitiva(matriz):
    n = len(matriz)  # Número de nodos
    for i in range(n):
        for j in range(n):
            if i != j and matriz[i][j]==1:
                for k in range(n):
                    if k != j and matriz[j][k]==1:
                        if matriz[i][k]==1:
                            return True
    return False


def determinar_tipo_grafo(matriz):
    reflexiva = es_reflexiva(matriz)
    simetrica = es_simetrica(matriz)
    antisimetrica = es_antisimetrica(matriz)
    transitiva = es_transitiva(matriz)

    print("\nPropiedades de la relación:")
    print(f"- Reflexiva: {'Sí' if reflexiva else 'No'}")
    print(f"- Simétrica: {'Sí' if simetrica else 'No'}")
    print(f"- Antisimétrica: {'Sí' if antisimetrica else 'No'}")
    print(f"- Transitiva: {'Sí' if transitiva else 'No'}")

    if reflexiva and simetrica and transitiva:
        print("\nEl grafo representa una RELACIÓN DE EQUIVALENCIA")
    else:
        print("\nNo es una relación de equivalencia")

    if reflexiva and antisimetrica and transitiva:
        print("El grafo representa una RELACIÓN DE ORDEN PARCIAL")
    else:
        print("No es una relación de orden parcial")

# --- REPRESENTACIÓN ASCII ---
def mostrar_ascii(m):
    print("\nRepresentación ASCII:")
    print("Nodos:", list(range(len(m))))
    print("Aristas:")
    for i in range(len(m)):
        for j in range(len(m)):
            if m[i][j]: print(f"{i}->{j}")

# ----------------------------
# Construcción de grafos y rutas
# ----------------------------
def generar_grafo_con_pesos(M):
    G = nx.Graph()  
    n = M.shape[0]
    for u in range(n):
        for v in range(n):
            if M[u,v]:
                w = random.randint(1,10)
                G.add_edge(u, v, weight=w)
    return G



def grafo_hamiltoniano(M, inicio, final):
    G = nx.Graph(M); G.remove_edges_from(nx.selfloop_edges(G))
    path, vis = [], set()
    def dfs(u):
        path.append(u); vis.add(u)
        if u == final and len(path) == G.number_of_nodes():
            return True
        for v in G.neighbors(u):
            if v not in vis and dfs(v):
                return True
        vis.remove(u); path.pop(); return False
    dfs(inicio)
    return G, path

def grafo_euler(M, inicio, final):
    G = nx.Graph(M); G.remove_edges_from(nx.selfloop_edges(G))
    path, vis = [], set()
    def dfs(u):
        path.append(u); vis.add(u)
        if u == final and len(path) == G.number_of_nodes():
            return True
        for v in G.neighbors(u):
            if v not in vis and dfs(v):
                return True
        vis.remove(u); path.pop(); return False
    ok = dfs(inicio)
    return G, (path if ok else [])

def grafo_greedy_undirected(Gw, inicio, final):
    current, visited, path, cost = inicio, {inicio}, [inicio], 0
    while current != final:
        cands = [(v, Gw[current][v]['weight']) for v in Gw.neighbors(current) if v not in visited]
        if not cands:
            break
        v,w = min(cands, key=lambda x: x[1])
        path.append(v); cost += w; visited.add(v); current = v
    return Gw, path, cost

def grafo_prim_lineal(Gw, inicio):
    n = Gw.number_of_nodes()
    visitados = set([inicio])
    camino = nx.Graph()
    camino.add_node(inicio)
    extremos = set([inicio])
    candidatos = []  # (peso, nodo_origen, nodo_destino)
    aristas_usadas = set()
    peso_total = 0
    
    # Función para agregar candidatos desde un extremo
    def agregar_candidatos(nodo):
        for vecino in Gw.neighbors(nodo):
            arista = tuple(sorted((nodo, vecino)))
            if arista not in aristas_usadas:
                peso = Gw[nodo][vecino]['weight']
                heapq.heappush(candidatos, (peso, nodo, vecino))
    
    # Inicializar con el nodo de inicio
    agregar_candidatos(inicio)
    
    while len(visitados) < n and candidatos:
        w, u, v = heapq.heappop(candidatos)
        arista = tuple(sorted((u, v)))
        
        # Validar si la arista ya fue usada
        if arista in aristas_usadas:
            continue
            
        # Validar si u sigue siendo extremo válido
        if u not in extremos:
            continue
            
        # Validar si v ya está en el camino
        if v in visitados:
            # Verificar que no se forme un ciclo
            if camino.degree(v) >= 2:
                continue  # No se puede conectar a nodos internos
        else:
            visitados.add(v)
        
        # Agregar arista al camino
        camino.add_edge(u, v, weight=w)
        aristas_usadas.add(arista)
        peso_total += w
        
        # Actualizar extremos
        extremos.discard(u)
        extremos.add(v)
        
        # Si u sigue siendo extremo (grado 1)
        if camino.degree(u) == 1:
            extremos.add(u)
        
        # Agregar nuevas opciones desde los extremos
        for extremo in list(extremos):
            agregar_candidatos(extremo)
    
    return Gw, camino, peso_total


def grafo_kruskal_lineal(G_w):
    parent = {u: u for u in G_w.nodes()}
    degree = {u: 0 for u in G_w.nodes()}
    edges = sorted(G_w.edges(data='weight'), key=lambda x: x[2])
    T = nx.Graph()
    total_weight = 0

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    for u, v, weight in edges:
        if degree[u] < 2 and degree[v] < 2:
            if find(u) != find(v):
                T.add_edge(u, v, weight=weight)
                union(u, v)
                degree[u] += 1
                degree[v] += 1
                total_weight += weight
        # Si ya se formó un camino con todos los nodos, salir
        if T.number_of_edges() == len(G_w.nodes()) - 1:
            break

    return (G_w, T, total_weight)

def grafo_color_ciclo(M):
    G = nx.Graph(M); G.remove_edges_from(nx.selfloop_edges(G))
    degrees = dict(G.degree())
    nodos = sorted(degrees, key=degrees.get, reverse=True)
    cmap, rem, col = {}, set(nodos), 0
    while rem:
        col += 1
        base = next(n for n in nodos if n in rem)
        cmap[base] = col; rem.remove(base)
        for u in nodos:
            if u in rem and all(cmap.get(nb)!=col for nb in G.neighbors(u)):
                cmap[u] = col; rem.remove(u)
    def backtrack(path, used):
        if len(path)==len(set(cmap.values())):
            return path+[path[0]] if G.has_edge(path[-1], path[0]) else None
        for nb in G.neighbors(path[-1]):
            c = cmap[nb]
            if nb not in path and c not in used:
                r = backtrack(path+[nb], used|{c})
                if r: return r
        return None
    cycle=[]
    for s in G.nodes():
        c = backtrack([s], {cmap[s]})
        if c:
            cycle=c; break
    return G, cmap, nodos, cycle

# ----------------------------
# Dibujo genérico con etiquetas de peso
# ----------------------------
def dibujar_en_ax(ax, G, node_color=None, elist=None, ecol='red', show_w=False, title='', directed_edges=True):
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=400, node_color=node_color)
    nx.draw_networkx_labels(G, pos, ax=ax)

    # Determine which edges to draw as base
    if directed_edges and G.is_directed():
        base_edges = list(G.edges())
    else:
        base_edges = [(u, v) for u, v in G.edges() if u != v]

    # Draw base edges lightly
    nx.draw_networkx_edges(G, pos, edgelist=base_edges, ax=ax, alpha=0.2)

    # Highlighted edges
    if elist:
        if directed_edges:
            nx.draw_networkx_edges(
                nx.DiGraph(elist), pos, ax=ax,
                edge_color=ecol, width=2,
                arrows=True, arrowstyle='-|>', arrowsize=15
            )
        else:
            nx.draw_networkx_edges(
                nx.DiGraph(elist), pos, ax=ax,
                edge_color=ecol, width=2
            )

    # Edge weight labels
    if show_w:
        labels = nx.get_edge_attributes(G, 'weight')
        draw_labels = {edge: labels[edge] for edge in base_edges if edge in labels}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=draw_labels, ax=ax,
            font_color='black', label_pos=0.5,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8, edgecolor='none')
        )

    ax.set_title(title)
    ax.axis('off')

# ----------------------------
# Función para realizar BFS ordenado
# ----------------------------
def grafo_bfs_ordenado(M, inicio):
    G = nx.Graph(M)
    G.remove_edges_from(nx.selfloop_edges(G))
    visited = {inicio}
    heap = [inicio]  # Usamos un heap para procesar el nodo más pequeño primero
    T = nx.DiGraph()
    
    while heap:
        u = heapq.heappop(heap)  # Extrae el nodo más pequeño
        # Ordena los vecinos numéricamente antes de procesarlos
        for v in sorted(G.neighbors(u)):
            if v not in visited:
                visited.add(v)
                heapq.heappush(heap, v)  # Agrega el vecino al heap
                T.add_edge(u, v)  # Añade la arista al árbol
    return G, T
# ----------------------------
# Función para realizar DFS
# ----------------------------
def grafo_dfs(M, inicio):
    G = nx.Graph(M)
    G.remove_edges_from(nx.selfloop_edges(G))
    n = G.number_of_nodes()
    
    if n == 0:
        return G, nx.DiGraph(), []  # Grafo vacío

    # Inicialización
    visited = set([inicio])
    path = [inicio]  # Camino completo (incluye retrocesos)
    T = nx.DiGraph()  # Árbol DFS
    T.add_node(inicio)
    
    # Pila: (nodo_actual, iterador_de_vecinos_ordenados)
    stack = [(inicio, iter(sorted(G.neighbors(inicio))))]
    
    while stack:
        u, neighbors_iter = stack[-1]
        try:
            # Obtener próximo vecino
            v = next(neighbors_iter)
            if v not in visited:
                visited.add(v)
                path.append(v)      # Registrar avance
                T.add_edge(u, v)    # Añadir arista al árbol
                
                # Apilar nuevo nodo con sus vecinos
                stack.append((v, iter(sorted(G.neighbors(v)))))
                
                # Detener si todos los nodos están visitados
                if len(visited) == n:
                    break
                    
        except StopIteration:
            # Retroceder: no hay más vecinos
            stack.pop()
            if stack:
                # Registrar retroceso al nodo anterior
                path.append(stack[-1][0])
    
    return G, T, path


# ----------------------------
# Main: mostrar con plt.show()
# ----------------------------
def main():
    m = int(input("Ingrese tamaño m: "))
    M = leer_matriz(m)
    # ... lectura y determinaciones previas ...

    # Generación de grafos y árboles
    i = int(input("Inicio : "))
    f = int(input("Fin : "))
    Gw = generar_grafo_con_pesos(M)
    G_bfs, T_bfs         = grafo_bfs_ordenado(M, i)
    G_dfs, T_dfs, path_dfs = grafo_dfs(M, i)
    # Otros grafos omitidos para brevedad

    # Lista de funciones de dibujo: ahora BFS y su árbol, DFS y su árbol
    funcs = [
        ("Grafo Dirigido", lambda ax: dibujar_en_ax(ax, nx.DiGraph(M), title="Grafo Dirigido")),
        # ... otras entradas ...
        ("BFS Ordenado", lambda ax: dibujar_en_ax(
            ax, G_bfs,
            elist=list(T_bfs.edges()),
            ecol='purple',
            title="BFS Ordenado"
        )),
        ("Árbol BFS", lambda ax: dibujar_en_ax(
            ax, T_bfs,
            node_color='lightgreen',
            elist=list(T_bfs.edges()),
            ecol='purple',
            directed_edges=True,
            title="Árbol BFS"
        )),
        ("DFS", lambda ax: dibujar_en_ax(
            ax, G_dfs,
            elist=list(zip(path_dfs, path_dfs[1:])),
            ecol='orange',
            title="DFS (Recorrido)"
        )),
        ("Árbol DFS", lambda ax: dibujar_en_ax(
            ax, T_dfs,
            node_color='lightcoral',
            elist=list(T_dfs.edges()),
            ecol='orange',
            directed_edges=True,
            title="Árbol DFS"
        )),
    ]

    # Ajuste de subplots: 2 filas x N columnas según funcs
    cols = len(funcs) // 2 + len(funcs) % 2
    fig, axes = plt.subplots(2, cols, figsize=(4*cols, 8))
    axes = axes.flatten()
    for idx, (_, fn) in enumerate(funcs):
        fn(axes[idx])
    # Ocultar ejes sobrantes
    for j in range(len(funcs), len(axes)):
        axes[j].axis('off')

    fig.tight_layout(pad=2.0)
    plt.show()


if __name__ == "__main__":
    main()
