# QA threat model

## Assets

- **Answer integrity:** answers should be supported by retrieved documentation and citations must resolve to real evidence chunks.
- **Provider quota:** untrusted input must not trigger unbounded model work or consume the per-model token allowance without limits.
- **Corpus integrity:** indexed documents and metadata should remain trustworthy; a poisoned document must not become executable instructions.

## Threats and mitigations

| Threat | Mitigations |
|---|---|
| Direct prompt injection in a question | Input length cap; frozen deterministic rules; a structured injection classifier runs only after the rules pass. A positive, invalid, or unavailable classifier result is refused. The router subsequently rejects out-of-scope questions. |
| Indirect injection in a retrieved chunk | Retrieved text is HTML-escaped and enclosed in `<untrusted_evidence>` delimiters. The synthesis instruction says evidence is data, not instructions. The verifier checks factual support and citations; the output gate rejects empty or uncited answers. Poisoned-chunk model testing remains a separate evaluation. |
| Resource exhaustion | Question length cap, retrieval concurrency semaphore, evidence token cap, answer token limit, cumulative graph token budget, bounded planner output, and approval interrupt when projected cost is high. |
| Unauthorized ingestion or corpus replacement | Ingestion endpoints require the configured `AEGIS_INGEST_TOKEN`; the token is not stored in the corpus. |

## Failure policy

The input classifier fails closed: malformed structured output, an exception, or a positive verdict produces a refusal before routing or retrieval. The deterministic rules short-circuit and refuse without calling the classifier. When answer verification fails or citations are invalid, the graph may revise at most twice; if it still cannot pass verification, it returns `insufficient evidence`. A final output gate independently refuses empty or uncited output.

## Residual risks

- Heuristics and a small LLM classifier can miss paraphrased or multilingual attacks. A classifier pass is not proof of safety.
- Delimiters and instructions reduce the chance that poisoned evidence changes model behavior; they do not guarantee isolation. The generator still sees the text.
- The verifier is an LLM judge, lightly calibrated at best. Its scores are indicative and all answer-quality results must be labeled **LLM-judged, uncalibrated** until human calibration exists.
- A malicious but valid document can contain convincing false claims. Citation validity does not establish source correctness.
- Token accounting and projected-cost estimates can differ from provider billing. Provider quotas can still reject an otherwise bounded request.
- The current corpus, development questions, and attack fixtures are small and not independently reviewed. Passing their tests does not establish production security.
