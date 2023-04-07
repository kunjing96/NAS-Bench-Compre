import time

from search_strategies import _register
from search_strategies.Base import Base


@_register
class Random(Base):

    def __call__(self):
        init_time = time.time()
        self.visited = []
        self.history  = []
        while len(self.history) < self.config.N:
            arch = self.search_space.sample()
            if arch not in self.visited and self.search_space.is_valid(arch):
                score, perf = self.estimation_strategy(arch)
                self.visited.append(arch)
                self.history.append({
                    'arch': arch,
                    'score': score,
                    'perf': perf,
                    'time': time.time()-init_time,
                })
        return max(self.history, key=lambda x: x['score']), self.history, time.time()-init_time
