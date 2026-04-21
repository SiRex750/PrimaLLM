from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel


class VerificationResult(BaseModel):
	is_verified: bool
	reason: str


def verify_claim(dirty_claim: str, clean_context: list[str], api_key: str) -> VerificationResult:
	premise = " ".join(clean_context).strip()

	model_name = "cross-encoder/nli-deberta-v3-small"
	tokenizer, model = _load_nli_model(model_name)
	inputs = tokenizer(
		premise,
		dirty_claim,
		return_tensors="pt",
		truncation=True,
		padding=True,
	)

	with torch.no_grad():
		outputs = model(**inputs)
		prediction = int(torch.argmax(outputs.logits, dim=-1).item())

	if prediction == 2:
		return VerificationResult(
			is_verified=True,
			reason="Verified by local DeBERTa-v3 NLI model: the claim is entailed by the context.",
		)

	if prediction == 0:
		reason = "Rejected by local DeBERTa-v3 NLI model: the claim contradicts the context."
	else:
		reason = "Rejected by local DeBERTa-v3 NLI model: the claim is not entailed by the context."

	return VerificationResult(is_verified=False, reason=reason)


@lru_cache(maxsize=1)
def _load_nli_model(model_name: str):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSequenceClassification.from_pretrained(model_name)
	model.eval()
	return tokenizer, model


if __name__ == "__main__":
	clean_context = [
		"Julius Caesar conquered Gaul.",
		"Caesar was assassinated by the Senate.",
	]

	claim_1 = "Caesar fought in Gaul."
	claim_1_result = verify_claim(claim_1, clean_context, "")
	print("Claim 1:", claim_1)
	print("is_verified:", claim_1_result.is_verified)
	print("reason:", claim_1_result.reason)

	claim_2 = "Caesar conquered Japan."
	claim_2_result = verify_claim(claim_2, clean_context, "")
	print("\nClaim 2:", claim_2)
	print("is_verified:", claim_2_result.is_verified)
	print("reason:", claim_2_result.reason)
