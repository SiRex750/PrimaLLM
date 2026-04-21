import tiktoken


class L1CacheManager:
	def __init__(self, max_tokens: int = 1000):
		self.max_tokens = max_tokens

		# Strict partition budgets (in tokens)
		self.system_budget = 200
		self.history_budget = 300
		self.facts_budget = 500

		# Partitioned cache stores
		self.system_prompt = {}
		self.chat_history = {}
		self.active_facts = {}

		self._encoding = tiktoken.get_encoding("o200k_base")

	def count_tokens(self, text: str) -> int:
		return len(self._encoding.encode(text))

	def _active_facts_token_usage(self) -> int:
		return sum(fact_data["tokens"] for fact_data in self.active_facts.values())

	def add_fact(self, subject: str, verb: str, object: str, pagerank_score: float):
		fact_text = f"{subject} {verb} {object}"
		fact_tokens = self.count_tokens(fact_text)

		if fact_tokens > self.facts_budget:
			raise ValueError("Fact exceeds the active_facts partition budget by itself.")

		# If full, keep evicting the lowest PageRank facts until there is room.
		while self.active_facts and (self._active_facts_token_usage() + fact_tokens) > self.facts_budget:
			lowest_fact = min(
				self.active_facts.items(),
				key=lambda item: item[1]["pagerank_score"],
			)[0]
			del self.active_facts[lowest_fact]

		self.active_facts[fact_text] = {
			"subject": subject,
			"verb": verb,
			"object": object,
			"pagerank_score": pagerank_score,
			"tokens": fact_tokens,
		}


if __name__ == "__main__":
	# Initialize cache but shrink the budget to 15 tokens to force an eviction
	cache = L1CacheManager()
	cache.facts_budget = 15

	print(f"Max Fact Tokens: {cache.facts_budget}")

	# Fact 1: High importance (Hub)
	print("\nAdding: Julius Caesar initiated civil war (Score: 0.8)")
	cache.add_fact("Julius Caesar", "initiated", "civil war", 0.8)

	# Fact 2: Medium importance
	print("Adding: Roman Republic replaced Roman Kingdom (Score: 0.5)")
	cache.add_fact("Roman Republic", "replaced", "Roman Kingdom", 0.5)

	print(f"Current Usage: {cache._active_facts_token_usage()} / 15 tokens")
	print(f"Current Facts in Cache: {list(cache.active_facts.keys())}")

	# Fact 3: Low importance (This should trigger an eviction of Fact 2!)
	print("\nAdding: Gaul is in Europe (Score: 0.1) -> EXPECTING EVICTION OF LOWEST SCORE")
	cache.add_fact("Gaul", "is in", "Europe", 0.1)

	print(f"New Usage: {cache._active_facts_token_usage()} / 15 tokens")
	print(f"Final Facts in Cache: {list(cache.active_facts.keys())}")
