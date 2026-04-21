import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class VerificationResult(BaseModel):
	is_verified: bool
	reason: str


def verify_claim(dirty_claim: str, clean_context: list[str], api_key: str) -> VerificationResult:
	client = OpenAI(api_key=api_key)

	context_text = "\n".join(f"- {item}" for item in clean_context)
	completion = client.beta.chat.completions.parse(
		model="gpt-4o-mini",
		messages=[
			{
				"role": "system",
				"content": (
					"You are a strict Natural Language Inference (NLI) evaluator. "
					"Decide whether the dirty_claim is logically supported by the clean_context only. "
					"Return is_verified=true only when the claim is directly supported by the context. "
					"If the claim contradicts the context or introduces new facts not present in the context, "
					"you must return is_verified=false. Provide a short reason."
				),
			},
			{
				"role": "user",
				"content": f"dirty_claim:\n{dirty_claim}\n\nclean_context:\n{context_text}",
			},
		],
		response_format=VerificationResult,
	)

	return completion.choices[0].message.parsed


if __name__ == "__main__":
	clean_context = [
		"Julius Caesar conquered Gaul.",
		"Caesar was assassinated by the Senate.",
	]
	load_dotenv()
	openai_api_key = os.getenv("OPENAI_API_KEY")
	if not openai_api_key:
		raise ValueError("OPENAI_API_KEY is not set. Add it to a local .env file or environment variables.")

	claim_1 = "Caesar fought in Gaul."
	claim_1_result = verify_claim(claim_1, clean_context, openai_api_key)
	print("Claim 1:", claim_1)
	print("is_verified:", claim_1_result.is_verified)
	print("reason:", claim_1_result.reason)

	claim_2 = "Caesar conquered Japan."
	claim_2_result = verify_claim(claim_2, clean_context, openai_api_key)
	print("\nClaim 2:", claim_2)
	print("is_verified:", claim_2_result.is_verified)
	print("reason:", claim_2_result.reason)
