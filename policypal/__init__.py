"""PolicyPal — three RAG architectures over an employee handbook.

Public API:

    from policypal import PolicyPal
    pal = PolicyPal("OmniCorp_Handbook_Challenge V2.pdf")
    ans = pal.answer("How many days of bereavement leave do I get?",
                     approach="traditional")   # | "pageindex" | "agentic"
    print(ans.render())

All three approaches share one parsing/citation/verification contract (see the
guide in rag_implementation_guide.md). The retrieval strategy is swappable; the
trust mechanism (mechanical quote verification) is not.
"""

from .pipeline import PolicyPal
from .contract import Answer, Claim, Citation

__all__ = ["PolicyPal", "Answer", "Claim", "Citation"]
