"""
Retrieval-augmented generation (RAG) for medical term definitions.

Grounds the "Medical terms" section of the explanation in trusted, locally stored
definitions instead of the language model's training data. Everything in this package
runs on the local machine and only ever sees REDACTED text.

Modules:
    build_kb   Parse the MedlinePlus health-topics XML into a local JSONL knowledge base.
    kb         Load the knowledge base (MedlinePlus + curated glossary) and index its names.
    terms      Find knowledge-base terms in redacted text (exact, alias and fuzzy matching).
"""
