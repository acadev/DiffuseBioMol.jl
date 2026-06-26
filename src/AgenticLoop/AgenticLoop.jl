"""
Phase 5: the resumable generate -> score -> triage -> curate -> retrain
controller (see Section 4 of the project plan). Deliberately a classical
closed-loop controller, not an LLM-agent orchestrator — the verifier head
(`Verification`) plays the role a planning agent would play in a more
speculative design. Must protect against reward hacking via a fixed replay
buffer from the original curated training set and a mandatory slice of
expensive external verification on every retrain cycle. Not yet implemented.
"""
module AgenticLoop

end # module
