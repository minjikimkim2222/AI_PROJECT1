import math
import numpy as np
import pandas as pd
import time
import random
import csv
import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

MUTATION_RATE = 0.001 #돌연변이 확률
SIZE = 1000 # 하나의 염색체(경로)에서 유전자(도시)의 개수
start = time.time()

#csv 파일 읽기로 도시의 x좌표, y좌표 읽기
def read_csv(file):
    data = pd.read_csv(filename, header=None, names=['x','y'])
    cities = np.array(data[['x', 'y']])
    return cities

#제안된 알고리즘의 유전자 클래스
class Chromosome_cluster:
    def __init__(self, g=[]):
        self.genes = g.copy() # 유전자는 리스트로 구현
        self.fitness = 0 # 적합도
        if self.genes.__len__() == 0: # 염색체가 초기 상태이면 초기화해, 유전 알고리즘에서 초기 해 설정
            i = 0
            self.genes.append(0)
            # 각 군집에서 dfs 시행
            min_path1 = dfs(cluster_list[2][0], [], DISTANCE, len(cluster_list[2]), cluster_list[2], 0)
            min_path2 = dfs(cluster_list[1][0], [], DISTANCE, len(cluster_list[1]), cluster_list[1], 0)
            min_path3 = dfs(cluster_list[0][0], [], DISTANCE, len(cluster_list[0]), cluster_list[0], 0)
            self.genes.extend(min_path1)
            self.genes.extend(min_path2)
            self.genes.extend(min_path3)
            
    def cal_fitness(self): # 적합도 계산 함수
        self.fitness = 0
        d = DISTANCE[0][self.genes[1]] #시작 도시 추가 
        for idx in range(1, SIZE-1):
            d += DISTANCE[self.genes[idx]][self.genes[idx+1]] # 각 도시들 사이 거리 더하기
        d += DISTANCE[self.genes[-1]][0] # 마지막 도시에서 시작 도시로 돌아오는 거리를 더한다.
        self.fitness = (1 / d)  # 거리가 짧을수록 적합도가 높아야 한다.
        return self.fitness
    
    def __str__(self):
        return self.genes.__str__()
            
def select(pop): #선택연산, population 리스트 내림차순 정렬 후, 가장 높은 적합도를 가진 2개의 도시 선택
    return pop[0], pop[1]

# 중복을 피하는 일점 교차 연산
def crossover(pop):
    father, mother = select(pop)
    index = random.randint(1, SIZE-2) # 교차 지점 선택
    # father 염색체를 복사해 교차로 받아온 유전자를 삭제한다.
    temp = father.genes.copy()
    for i in mother.genes[index:]:
        temp.remove(i)
    child1 = temp + mother.genes[index:] # father 염색체 중 남은 유전자를 교차로 받아온 유전자와 합친다.
    temp = mother.genes.copy() # mother 염색체를 복사해 교차로 받아온 유전자를 삭제한다.
    for i in father.genes[index:]: 
        temp.remove(i)
    child2 = temp + father.genes[index:] # mother 염색체 중 남은 유전자를 교차로 받아온 유전자와 합친다.
    return (child1, child2)

# mutual swap 돌연변이 연산
def mutate(c):
    for i in range(1, SIZE):
        if random.random() < MUTATION_RATE:
            g = random.randint(1, SIZE-1) # 시작도시를 제외한 도시 하나를 선택
            while g == i:
                g = random.randint(1, SIZE-1)
            c.genes[i], c.genes[g] = c.genes[g], c.genes[i] # 두 도시를 상호 교환

#각 도시들 사이 거리를 계산해 distance_list에 저장
def calculate_distances(points):
    n = len(points)
    distance_list = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            distance = ((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)**0.5
            distance_list[i][j] = distance
            distance_list[j][i] = distance
    return distance_list

#dfs함수를 통해 1000개의 도시를 방문하는 tsp문제를 푼다.
#cur : 현재 방문한 도시, visited : 방문한 도시 목록, cost : 현재까지의 비용
#graph : 도시 간의 거리 정보, n : 도시 개수, list_rd : 무작위로 선정된 도시 배열, i : 현재 도시 순번
def dfs(cur, visited, graph, n, list_rd, i):
        min_path = None
        if len(visited) != n: # 재귀 탈출 : 모든 도시를 방문했을 때
            next_path = dfs(list_rd[i], visited + [list_rd[i]], graph, n, list_rd, i+1)
        else:  # 시작 도시를 거리와 경로에 추가
            next_path = visited
        min_path = next_path
        return min_path #최소 경로와 비용 반환

# 메인 프로그램
filename = '2023_AI_TSP.csv'
temp = read_csv(filename)
DISTANCE = calculate_distances(temp)
population = []
min_length = float('inf')
i=0

#데이터 클러스터링
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10)
y_pred = kmeans.fit_predict(temp)
# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed clusters
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
silhouette_avg = silhouette_score(temp, y_pred)
# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(temp, y_pred)
# plot
plt.scatter(temp[:,0], temp[:,1])
plt.savefig('05_kmeans_original.png')
plt.clf()
plt.scatter(temp[:,0], temp[:,1], c=y_pred)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
plt.savefig('05_kmeans_centers.png')
cluster_list = [[] for i in range(n_clusters)]
for i in range(1, len(y_pred)):
    cluster_list[y_pred[i]].append(i)
if not y_pred[0] == 0:
    cluster_list[y_pred[0]], cluster_list[0] = cluster_list[0], cluster_list[y_pred[0]]

# 초기 염색체를 생성하여 객체 집단에 추가한다.
i = 0
population_size = 10
while i<population_size:
    population.append(Chromosome_cluster())
    i += 1
count=0
population.sort(key=lambda x: x.cal_fitness(), reverse=True)
count=1

# sorting을 하기 때문에 [0]을 비교
while population[0].cal_fitness()  :
    new_pop = []
    # 선택과 교차 연산
    for _ in range(population_size//2):
        c1, c2 = crossover(population)
        new_pop.append(Chromosome_cluster(c1))
        new_pop.append(Chromosome_cluster(c2))
    #최소 길이
    length = 1 / population[0].cal_fitness()
    if length < min_length:
        min_length = length
    # 자식 세대가 부모 세대를 대체한다.    
    # 깊은 복사를 수행한다.
    population = new_pop.copy();
    # 돌연변이 연산
    for c in population: mutate(c)
    # 출력을 위한 정렬
    population.sort(key=lambda x: x.cal_fitness(), reverse=True)
    now = time.time()
    if count%10 == 0:
        print("Generation :", count)
        print("Population Size :", population_size)
        print("Best Length :", round(min_length,2))
        print("Time Elapsed :", round(now - start, 2))
        print("Total Node List :", population[0],"\n")
    population_size = round(pow(210000 /length,3.5) * 0.003)
    count += 1
    if count == 100000:
        break

#csv 파일 경로 저장
path = population[0].genes
f = open('example_solution.csv', 'w', newline='')
wr = csv.writer(f)
for i in range(len(path)):
    wr.writerow([path[i]])
f.close()
