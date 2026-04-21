import os

from dotenv import load_dotenv
from openai import OpenAI

from caveman import extract_knowledge_graph
from memory_manager import L1CacheManager
from verification_layer import verify_claim
from wiki_storage import load_wiki, save_verified_fact


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise ValueError("OPENAI_API_KEY is not set. Add it to a local .env file or environment variables.")

client = OpenAI(api_key=api_key)

l1_cache = L1CacheManager()

wiki_facts = load_wiki()
for fact in wiki_facts:
	l1_cache.add_fact(
		fact["subject"],
		fact["verb"],
		fact["object"],
		pagerank_score=0.5,
	)


def build_active_facts_system_prompt(cache: L1CacheManager) -> str:
	if not cache.active_facts:
		return "No active facts available."

	fact_lines = [f"- {fact_text}" for fact_text in cache.active_facts.keys()]
	return "Active facts:\n" + "\n".join(fact_lines)


if __name__ == "__main__":
	while True:
		user_input = input("User: ")
		if user_input.strip().lower() == "exit":
			break

		system_prompt = build_active_facts_system_prompt(l1_cache)
		completion = client.chat.completions.create(
			model="gpt-4o-mini",
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_input},
			],
		)

		ai_response = completion.choices[0].message.content or ""
		print(f"AI: {ai_response}")

		dirty_graph = extract_knowledge_graph(ai_response, api_key)
		for triple in dirty_graph.triples:
			dirty_claim = f"{triple.subject} {triple.verb} {triple.object}"
			clean_context = list(l1_cache.active_facts.keys())
			verification = verify_claim(dirty_claim, clean_context, api_key)

			if verification.is_verified:
				print(f"Verified: {dirty_claim} | Reason: {verification.reason}")
				save_verified_fact(triple.subject, triple.verb, triple.object)
				l1_cache.add_fact(triple.subject, triple.verb, triple.object, pagerank_score=0.5)
			else:
				print(f"Rejected: {dirty_claim} | Reason: {verification.reason}")
