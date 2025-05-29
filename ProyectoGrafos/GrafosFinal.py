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
# Hamilton buscando el camino más largo
def grafo_hamiltoniano(M, inicio, final):
    G = nx.Graph(M)
    G.remove_edges_from(nx.selfloop_edges(G))
    mejor_camino = []
    path, vis = [], set()

    def dfs(u):
        nonlocal mejor_camino
        path.append(u)
        vis.add(u)
        if u == final:
            if len(path) > len(mejor_camino):
                mejor_camino = path.copy()
        for v in G.neighbors(u):
            if v not in vis:
                dfs(v)
        vis.remove(u)
        path.pop()

    dfs(inicio)
    return G, mejor_camino

#Hamilton tocando todos los nodos

# def grafo_hamiltoniano(M, inicio, final):
#    G = nx.Graph(M); G.remove_edges_from(nx.selfloop_edges(G))
#    path, vis = [], set()
#    def dfs(u):
#        path.append(u); vis.add(u)
#        if u == final and len(path) == G.number_of_nodes():
#            return True
#        for v in G.neighbors(u):
#            if v not in vis and dfs(v):
#                return True
#        vis.remove(u); path.pop(); return False
#    dfs(inicio)
#    return G, path
# Euler 
# def grafo_euler(M, inicio, final):
#     G = nx.Graph(M); G.remove_edges_from(nx.selfloop_edges(G))
#     path, vis = [], set()
#     def dfs(u):
#         path.append(u); vis.add(u)
#         if u == final and len(path) == G.number_of_nodes():
#             return True
#         for v in G.neighbors(u):
#             if v not in vis and dfs(v):
#                 return True
#         vis.remove(u); path.pop(); return False
#     ok = dfs(inicio)
#     return G, (path if ok else [])

# Euler sin conectar todos los nodos
def grafo_euler(M, inicio, final):
    G = nx.Graph(M)
    G.remove_edges_from(nx.selfloop_edges(G))
    used = set()
    path = []

    def dfs(u):
        for v in list(G.neighbors(u)):
            e = tuple(sorted((u, v)))
            if e not in used:
                used.add(e)
                dfs(v)
        path.append(u)

    dfs(inicio)
    path = path[::-1]
    # Ajustar para terminar en 'final' si es posible
    if path and path[-1] != final:
        # Intentar rotar el camino para terminar en 'final'
        if final in path:
            idx = path.index(final)
            path = path[:idx+1]
        else:
            path = []
    return G, path
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
    camino = [inicio]
    peso_total = 0
    actual = inicio

    while len(visitados) < n:
        # Buscar el vecino no visitado más cercano
        vecinos = [(v, Gw[actual][v]['weight']) for v in Gw.neighbors(actual) if v not in visitados]
        if not vecinos:
            break  # No hay más vecinos no visitados, termina la ruta
        # Elegir el vecino con menor peso
        siguiente, peso = min(vecinos, key=lambda x: x[1])
        camino.append(siguiente)
        peso_total += peso
        visitados.add(siguiente)
        actual = siguiente

    # Crear grafo camino
    G_camino = nx.Graph()
    for i in range(len(camino) - 1):
        u, v = camino[i], camino[i+1]
        G_camino.add_edge(u, v, weight=Gw[u][v]['weight'])

    return Gw, G_camino, peso_total

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
                nx.Graph(elist), pos, ax=ax,
                edge_color=ecol, width=2,
                arrows=True, arrowstyle='-|>', arrowsize=15
            )
        else:
            nx.draw_networkx_edges(
                nx.Graph(elist), pos, ax=ax,
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
# Dibujo de árbol en diagrama jerárquico
# ----------------------------
def dibujar_diagrama_arbol(ax,T,root,title,ecol,node_color):
    try: pos=nx.nx_agraph.graphviz_layout(T,prog='dot')
    except: pos=nx.spring_layout(T,seed=42)
    levels=nx.single_source_shortest_path_length(T,root)
    maxl=max(levels.values())
    ys={n:1-(levels[n]/maxl if maxl>0 else 0) for n in T.nodes()}
    xs={n:pos[n][0] for n in T.nodes()}
    pos={n:(xs[n],ys[n]) for n in T.nodes()}
    nx.draw_networkx_edges(T,pos,ax=ax,edgelist=T.edges(),edge_color=ecol,arrows=True)
    nx.draw_networkx_nodes(T,pos,ax=ax,node_size=600,node_color=node_color)
    nx.draw_networkx_labels(T,pos,ax=ax)
    xmin=min(x for x,y in pos.values())
    for lvl in range(maxl+1):
        yv=1-(lvl/maxl if maxl>0 else 0)
        ax.text(xmin-50,yv,f"NIVEL {lvl}",va='center',fontsize=10)
    ax.set_title(title); ax.axis('off')


# ----------------------------
# Main: mostrar con plt.show()
# ----------------------------
def main():
    m = int(input("Ingrese tamaño m: "))
    M = leer_matriz(m)
    determinar_tipo_grafo(M)
    mostrar_ascii(M)
    i = int(input("Inicio : "))
    f = int(input("Fin : "))
    
    Gw = generar_grafo_con_pesos(M)
    G_bfs, T_bfs         = grafo_bfs_ordenado(M, i)
    G_h, path_h          = grafo_hamiltoniano(M, i, f)
    G_e, path_e          = grafo_euler(M, i, f)
    G_g, path_g, cost_g  = grafo_greedy_undirected(Gw, i, f)
    G_prim, T_prim, w_prim = grafo_prim_lineal(Gw, i)
    G_k, T_k, w_kruskal  = grafo_kruskal_lineal(Gw)
    G_c, cmap, ordn, cycle_c = grafo_color_ciclo(M)
    G_dfs, T_dfs, path_dfs = grafo_dfs(M, i)
    funcs = [
        ("Grafo Dirigido", lambda ax: dibujar_en_ax(ax, nx.DiGraph(M), show_w=False, title="Grafo Dirigido")),
        ("Coloración & Ciclo", lambda ax: dibujar_en_ax(
            ax, G_c,
            node_color=[cmap[n] for n in G_c.nodes()],
            elist=list(zip(cycle_c, cycle_c[1:])) if cycle_c else None,
            show_w=False,
            title=f"Coloración & ciclo óptimo:\n {'->'.join(str(cmap[n]) for n in ordn)}"
        )),
        ("Euler", lambda ax: dibujar_en_ax(
            ax, G_e,
            elist=list(zip(path_e, path_e[1:])) if path_e else None,
            title=f"Euler: \n{'->'.join(str(n) for n in path_e)}" if path_e else "Euler — No posible"
        )),
        ("Hamiltoniano", lambda ax: dibujar_en_ax(
            ax, G_h,
            elist=list(zip(path_h, path_h[1:])) if path_h else None,
            title=f"Hamilton: \n{'->'.join(str(n) for n in path_h)}" if path_h else "Hamiltoniano — No posible"
        )),
        ("Dijkstra", lambda ax: dibujar_en_ax(
            ax, G_g,
            elist=list(zip(path_g, path_g[1:])) if path_g else None,
            show_w=True,
            title=f"Dijkstra (peso {cost_g}): \n{'->'.join(str(n) for n in path_g)}" if path_g else "Dijkstra — No posible"
        )),
        ("Prim", lambda ax: dibujar_en_ax(
            ax, G_prim,
            elist=list(T_prim.edges()),
            show_w=True,
            ecol='blue',
            title=f"Prim (peso {w_prim}):\n{'->'.join(f'{u}-{v}' for u,v in T_prim.edges())}"
        )),
        ("Kruskal", lambda ax: dibujar_en_ax(
            ax, G_k,
            elist=list(T_k.edges()) if T_k.edges() else None,
            show_w=True,
            ecol='green',
            directed_edges=False,
            title=f"Kruskal (peso {w_kruskal}):\n{'->'.join(f'{u}-{v}' for u,v in T_k.edges())}"
        )),
        ("BFS Recorrido", lambda ax: dibujar_en_ax(ax,G_bfs,elist=list(T_bfs.edges()),ecol='purple',
            title="Anchura")),
        ("Árbol BFS", lambda ax: dibujar_diagrama_arbol(ax,T_bfs,i,"Árbol Anchura",ecol='purple',
            node_color='lightgreen')),
        ("DFS Recorrido", lambda ax: dibujar_en_ax(ax,G_dfs,elist=list(T_dfs.edges()),ecol='orange',
            title="Profundidad")),
        ("Árbol DFS", lambda ax: dibujar_diagrama_arbol(ax,T_dfs,i,"Árbol Anchura",ecol='orange',
            node_color='lightcoral'))
    ]

# Determinar filas y columnas para acomodar todos los subplots
    total = len(funcs)
    cols = total // 2 + total % 2
    rows = 2
    # Ajustar figsize: más ancho y menos alto para mejor proporción
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 3.5*rows))
    axes_flat = axes.flatten()
    for idx, (_, fn) in enumerate(funcs):
        fn(axes_flat[idx])
    # Ocultar celdas sin usar
    for j in range(len(funcs), len(axes_flat)):
        axes_flat[j].axis('off')
    # Ajustar márgenes para aprovechar al máximo
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.07,
                        wspace=0.4, hspace=0.4)
    plt.show()

if __name__=="__main__":
    main()
