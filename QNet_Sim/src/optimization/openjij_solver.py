import openjij as oj

def solve_sa(bqm, num_reads, seed=None):
    sampler = oj.SASampler()
    return sampler.sample(bqm, num_reads=num_reads, seed=seed)

def solve_sqa(bqm, num_reads, seed=None):
    sampler = oj.SQASampler()
    return sampler.sample(bqm, num_reads=num_reads, seed=seed)
