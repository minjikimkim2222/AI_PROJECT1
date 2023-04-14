import math
import numpy as np
import pandas as pd
import time
import random
import csv

POPULATION_SIZE = 300 #개체 크기
MUTATION_RATE = 0.001 #돌연변이 확률
SIZE = 1000 # 하나의 염색체(경로)에서 유전자(도시)의 개수

start = time.time()

#염색체 클래스
class Chromosome:
    def __init__(self, g=[]): #유전자(도시) 리스트와 적합도를 초기화하는 함수
        self.genes = g.copy()
        self.fitness = 0 # 적합도
        if self.genes.__len__() == 0: # 염색체가 초기 상태이면 초기화해, 유전 알고리즘에서 초기 해 설정
            i = 0
            self.genes.append(0)
            while i < SIZE-1: # while문을 돌며 유전자(도시) 리스트 채우기, i : 현재 추가한 도시 개수
                num = random.randint(1, SIZE-1)
                while num in self.genes: # 겹치는 node 있으면 다시 뽑는다.
                    num = random.randint(1, SIZE-1)
                self.genes.append(num)
                i += 1
    
    def cal_fitness(self): # 적합도 계산 함수
        self.fitness = 0
        d = DISTANCE[0][self.genes[1]] # 시작 도시 추가
        for idx in range(1, SIZE-1):
            d += DISTANCE[self.genes[idx]][self.genes[idx+1]] # 각 도시들 사이의 거리를 더한다.
        d += DISTANCE[self.genes[-1]][0] # 마지막 도시에서 시작 도시로 돌아오는 거리를 더한다.
        self.fitness = (1 / d)  # 거리가 짧을수록 적합도가 높아야 하기에, 거리의 역수로 적합도 계산
        return self.fitness
    
    def __str__(self):
        return self.genes.__str__()

#시도했던 첫번째 선택연산 -> 룰렛 휠 연산
# max_value = sum([c.cal_fitness() for c in pop]) # 전체 개체의 적합도의 합
# pick = random.uniform(0, max_value) # 0~max_value 사이의 랜덤 실수를 리턴
# current = 0
# # 룰렛휠에서 어떤 조각에 속하는지를 알아내는 루프
# for c in pop:
#     current += c.cal_fitness()
#     if current > pick:
#         return c

def select(pop): #두 번째 선택연산, population 리스트 내림차순 정렬 후, 가장 높은 적합도를 가진 2개의 도시 선택
    return pop[0], pop[1]

# 중복을 피하는 일점 교차 연산
def crossover(pop):
    father, mother = select(pop)
    index = random.randint(1, SIZE-2) # 교차 지점 선택
    temp = father.genes.copy() # father 염색체를 복사해 교차로 받아온 유전자를 삭제
    for i in mother.genes[index:]:
        temp.remove(i)
    child1 = temp + mother.genes[index:] # father 염색체 중 남은 유전자를 교차로 받아온 유전자와 합친다.
    temp = mother.genes.copy() # mother 염색체를 복사해 교차로 받아온 유전자를 삭제
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

#csv 파일 읽기로 도시의 x, y좌표 읽기
def read_csv(file):
    data = pd.read_csv(filename, header=None, names=['x','y'])
    cities = np.array(data[['x', 'y']])
    return cities

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

# 메인 프로그램
filename = '2023_AI_TSP.csv'
temp = read_csv(filename)
DISTANCE = calculate_distances(temp)
population = []
min_length = float('inf')
i=0

# 초기 염색체를 생성하여 객체 집단에 추가한다.
while i<POPULATION_SIZE:
    population.append(Chromosome())
    i += 1
count=0
population.sort(key=lambda x: x.cal_fitness(), reverse=True)
count=1

while population[0].cal_fitness()  : # sorting을 하기 때문에 [0]을 비교
    new_pop = []
    
    # 선택과 교차 연산
    for _ in range(POPULATION_SIZE//2):
        c1, c2 = crossover(population)
        new_pop.append(Chromosome(c1))
        new_pop.append(Chromosome(c2))
        
    #최소 길이
    length = 1 / population[0].cal_fitness()
    if length < min_length: #최소 길이 갱신
        min_length = length
    # 자식 세대가 부모 세대로 교체
    population = new_pop.copy();
    
    # 돌연변이 연산
    for c in population: mutate(c)
    
    # 정렬을 한 뒤, 출력
    population.sort(key=lambda x: x.cal_fitness(), reverse=True)
    now = time.time()
    if count%10 == 0:
        print("Generation : ", count)
        print("Best Length :  ", round(min_length,2))
        print("Time Elapsed : ", round(now - start, 2), "\n")
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
