import random
import matplotlib.pyplot as plt
import sys
import csv
import numpy as np
import pandas as pd
import random as rd
import time
from itertools import permutations

start = time.time()

# read_csv함수를 통해 csv파일에서 도시 x, y좌표를 가져온다
def read_csv(filename):
    data = pd.read_csv(filename, header=None, names=['x','y'])
    cities = np.array(data[['x', 'y']])
    return cities

#완전 무작위 탐색을 여러 번 반복해 최소 경로 찾기
#cur : 현재 방문한 도시, visited : 방문한 도시 목록, cost : 현재까지의 비용
#graph : 도시 간의 거리 정보, n : 도시 개수, list_rd : 무작위로 선정된 도시 배열, i : 현재 도시 순번
def follow(cur, visited, cost, graph, n, list_rd, i):
    min_cost = float('inf')
    min_path = None
    next_cost = cost + graph[cur][list_rd[i]]
    if len(visited) == n-1:  # 재귀 탈출 : 모든 도시를 방문했을 때
        next_cost += graph[list_rd[i]][0] # 시작 도시를 거리와 경로에 추가
        next_path = visited + [list_rd[i], 0]
    else: 
        next_path, next_cost = follow(list_rd[i], visited + [list_rd[i]], next_cost, graph, n, list_rd, i+1)
    if next_cost < min_cost:
        min_cost = next_cost
        min_path = next_path
    return min_path, min_cost #최소 경로와 비용 반환

# 도시 수
n = 1000

# 도시 자료 가져오기
filename = '2023_AI_TSP.csv'
cities = read_csv(filename)


# 각 도시 간의 거리 정보를 graph 리스트에 저장
graph = [[0] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i != j: # 같은 도시가 아닐 때
            x1, y1 = cities[i]
            x2, y2 = cities[j]
            # 유클리드 거리 계산
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            graph[i][j] = dist

min_cost = float('inf')
min_path = None
li = list(range(1,n))
list_rd = []



#무작위 반복 횟수
rand_num = 100000

for i in range(rand_num): # 무작위 선정을 100번 반복
    samplelist = rd.sample(li,n-1)
    list_rd.append(samplelist)
    #follow함수 호출해 최소 경로와 비용을 path와 cost에 각각 반환
    path, cost = follow(0, [0], 0, graph, n, list_rd[i], 0)
    if cost < min_cost: #최소 비용과 경로 갱신
        min_cost = cost
        min_path = path
    now = time.time()
    if (i%100 == 0): # 100번째마다 현재 상태 출력
        print('\nAttemp:',i)
        print('Best Length:',round(min_cost,2))
        print('Time Elapsed:', round(now-start,2))
    
    if min_cost < 30000: #최소 비용이 30000보다 작으면 break(완전 무작위 탐색이 4만대가 나오는지 체크)
        break

cost = min_cost #최종 비용
path = min_path #최종 경로

print(min_path)
print(min_cost)

# x좌표, y좌표 시각화
x = [city[0] for city in cities]
y = [city[1] for city in cities]
plt.scatter(x, y)

#csv 파일에 최적의 경로 저장
f = open('example_solution.csv', 'w', newline='')
wr = csv.writer(f)
for i in range(len(path)-1):
    wr.writerow([path[i]])
f.close()

# 최단 경로 시각화
for i in range(len(path)-1):
    plt.plot([x[path[i]], x[path[i+1]]], [y[path[i]], y[path[i+1]]], 'red', linewidth=1)
plt.show()
