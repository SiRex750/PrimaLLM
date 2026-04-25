import sys
from caveman.core.cache import L1Cache, MIN_FACTS_FLOOR
from shared.triple import KnowledgeTriple

print(f"DEBUG: MIN_FACTS_FLOOR is {MIN_FACTS_FLOOR}")

# Force a tiny budget that is exceeded by a single triple
cache = L1Cache(budgets={'facts': 1})

for i in range(5):
    t = KnowledgeTriple(f'entity{i}', 'relates to', f'object{i}')
    cache.add_fact(t)
    print(f"Added triple {i}, count now {len(cache.set_facts)}")

print(f'Facts remaining: {len(cache.set_facts)}')
assert len(cache.set_facts) == MIN_FACTS_FLOOR, f'Expected {MIN_FACTS_FLOOR}, got {len(cache.set_facts)}'
print('PASS: floor enforced correctly at minimum')
