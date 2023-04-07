import time
import random
from functools import cmp_to_key

from search_strategies import _register
from search_strategies.Base import Base


@_register
class AgingEvolution(Base):

    def cmp(self, x, y):
        ret = x['score'] - y['score']
        if ret<0: return -1
        elif ret>0: return 1
        else: return 0

    def mutate(self, arch):
        print('mutation ......')
        if random.random() < self.config.MATUTEPROB:
            decoded_arch = self.search_space.decode(arch)
            k = random.choices(self.search_space.choices.keys())
            if k in ['depth']:
                new_depth = random.choice(self.search_space.choices[k])
                if new_depth > decoded_arch[k]:
                    decoded_arch['mlp_ratio'] = decoded_arch['mlp_ratio'] + [random.choice(self.search_space.choices['mlp_ratio']) for _ in range(new_depth - decoded_arch[k])]
                    decoded_arch['num_heads'] = decoded_arch['num_heads'] + [random.choice(self.search_space.choices['num_heads']) for _ in range(new_depth - decoded_arch[k])]
                else:
                    decoded_arch['mlp_ratio'] = decoded_arch['mlp_ratio'][:new_depth]
                    decoded_arch['num_heads'] = decoded_arch['num_heads'][:new_depth]
                decoded_arch['depth'] = new_depth
            elif isinstance(decoded_arch[k], list) and decoded_arch['depth'] and len(decoded_arch[k]) == decoded_arch['depth']:
                i = random.choices(range(decoded_arch['depth']))
                decoded_arch[k][i] = random.choice(self.search_space.choices[k])
            else:
                decoded_arch[k] = random.choice(self.search_space.choices[k])
            return self.search_space.encode(decoded_arch)

    def next_generation(self, population):
        next_generation = []
        for _ in range(self.config.PATIENCEFACTOR):
            samples  = random.sample(population, self.config.TOURNAMENTSIZE)
            parents = [p['arch'] for p in sorted(samples, key=cmp_to_key(self.cmp), reverse=True)[:self.config.NUMPARENTS]]
            for parent in parents:
                for _ in range(int(self.config.NUMUTATES / self.config.NUMPARENTS)):
                    child = self.mutate(parent)
                    if child not in self.visited and self.search_space.is_valid(child):
                        next_generation.append(child)
                        self.visited.append(child)
                    if len(next_generation) >= self.config.NUMUTATES:
                        return next_generation
        return next_generation

    def __call__(self):
        init_time = time.time()
        self.population = []
        self.visited = []
        self.history  = []
        while len(self.population) < self.config.NUMPOPULATION:
            arch = self.search_space.sample()
            if arch not in self.visited:
                score, perf = self.estimation_strategy(arch)
                self.population.append({
                    'arch': arch,
                    'score': score,
                })
                self.visited.append(arch)
                self.history.append({
                    'arch': arch,
                    'score': score,
                    'perf': perf,
                    'time': time.time()-init_time,
                })
        while len(self.history) < self.config.N:
            next_generation = self.next_generation(self.population)
            for arch in next_generation:
                score, perf = self.estimation_strategy(arch)
                self.population.append({
                    'arch': arch,
                    'score': score,
                })
                del self.population[0]
                self.history.append({
                    'arch': arch,
                    'score': score,
                    'perf': perf,
                    'time': time.time()-init_time,
                })
        return max(self.history, key=lambda x: x['score']), self.history, time.time()-init_time
