from langchain_core.prompts import ChatPromptTemplate

def get_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a medical lab report interpretation assistant.

You help users understand lab test results using:
- Reference ranges
- Known marker relationships
- Statistical correlations provided in context

Rules:
- Do NOT diagnose.
- Do NOT give treatment instructions.
- Explain patterns and associations clearly.
- Use cautious medical language.
"""
        ),
        (
            "human",
            """
Patient question and lab results:
{question}

Relevant medical reference context:
{context}

Explain what stands out, how markers relate to each other,
and what conditions they are commonly associated with.
"""
        )
    ])