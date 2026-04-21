import json
import os


def save_verified_fact(subject: str, verb: str, object: str, filename: str = "karpathy_wiki.json"):
	if os.path.exists(filename):
		with open(filename, "r", encoding="utf-8") as file:
			facts = json.load(file)
	else:
		facts = []

	facts.append({"subject": subject, "verb": verb, "object": object})

	with open(filename, "w", encoding="utf-8") as file:
		json.dump(facts, file, indent=4)


def load_wiki(filename: str = "karpathy_wiki.json") -> list:
	if os.path.exists(filename):
		with open(filename, "r", encoding="utf-8") as file:
			return json.load(file)

	return []


if __name__ == "__main__":
	print("Saving mock verified fact...")
	save_verified_fact("Augustus", "founded", "Roman Empire")

	print("Loading wiki data...")
	wiki_facts = load_wiki()
	print("Loaded facts:")
	print(wiki_facts)
