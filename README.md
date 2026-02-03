# OptimizingWhatWeTrust

This repository provides the **exact multi-agent prompting configuration** used to reproduce the framing annotation and arbitration experiments reported in the paper.

---

## Models (exact)

```text
LABELER_A_MODEL = "qwen2.5:3b-instruct-q4_K_M"
LABELER_B_MODEL = "mistral:7b-instruct-q4_K_M"
CRITIC_MODEL    = "gemma2:9b-instruct-q4_K_M"
```

---

## Labels / Taxonomy (exact)

```text
Security/Threat
Moral/Religious
Economic/Cost–Benefit
Identity/Group
Rights/Justice
Public Health/Safety
Uncertain
```

---

## Prompts

### Labeler — System Prompt (exact)

```text
You label FRAMING in short texts using a fixed taxonomy. 
Texts may be Arabic or English. 
Base your decision on evidence in the text(s) and return STRICT JSON only. 
If no frame clearly fits, return 'Uncertain'. Start with '{' and end with '}'. 
Use DOUBLE quotes only. No prose.
```

---

### Labeler — User Prompt Template (exact)

```text
Task: Assign the single most dominant FRAME for how the text frames its content.

Text:
{text}

Frames taxonomy (choose exactly one; use 'Uncertain' only if no frame fits):
{TAXONOMY}

Return JSON ONLY:
{
 "label": ["<exactly one from taxonomy>"],
 "confidence": <0..1>,
 "rationale": "<<=40 words>",
 "evidence_span": "<exact phrase(s) from the text>"
}
```

---

### Critic / Judge — System Prompt (exact)

```text
You are an arbiter. Compare two labeler JSONs and decide a final label using a rubric. Return STRICT JSON only, no prose. Start with '{' and end with '}'.
```

---

### Critic / Judge — User Prompt Template (exact)

```text
Text:
{text}

LabelerA JSON:
{json.dumps(ja, ensure_ascii=False)}

LabelerB JSON:
{json.dumps(jb, ensure_ascii=False)}

Rubric (score 0–2 each; sum 0–8):
R1 Evidence quality
R2 Taxonomy fit
R3 Clarity/non-contradiction
R4 Justification sufficiency

Return FINAL JSON ONLY:
{
 "final_label": ["..."],
 "critic_score_sum": 0,
 "final_confidence": 0.0,
 "final_rationale": ""
}
```

---

**Reproducibility note:** All agent outputs must be **STRICT JSON only**, and all labels must **exactly match** the taxonomy strings above.
