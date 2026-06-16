# Don't ReadMe

## PolicyPal — Personal Notes & Commentary

> A companion to [`README.md`](README.md) and [`TECHNICAL_GUIDE.md`](TECHNICAL_GUIDE.md).
> These are opinions and field notes, not formal documentation — the *why* and the
> *what I'd watch out for*, rather than the *what* and the *how*.

---

## Summary

- The core idea is to get the most out of the **cheapest models and the fewest tokens** — maximize output quality per dollar.
- The flip side is real: **fewer input tokens and cheaper models generally mean lower-quality results.** Expect that trade-off; don't be surprised by it.
- In this project `top_k` is set to **6** — only the 6 most *similar* chunks, and *similar* is not the same as *relevant*. On my own knowledge base I started at `top_k = 10`, but I kept finding that the 11th or 12th result was often more relevant to the query, so I've used **12** since.

---

## Notes on the three approaches

### Traditional RAG

- **Lacks structure.** Chunking cuts sentences and paragraphs mid-thought. Token overlap helps, but the span you actually need may still fall outside the chunks that survive the similarity filter.
- **Don't use it for structural / multi-faceted questions** — anything involving **steps, processes, or enumeration.** It's weak there: answers can be correct but partial, missing key information. For example:
  - "What are the steps in establishing an investment policy statement (IPS)?"
  - "The characteristics of passive vs. active investment strategies, and their applications for a casualty insurance company versus a defined-benefit pension plan."

### PageIndex

- It has a real-world analogue that has organized human problem-solving for thousands of years: **bureaucracy** — divide the problem into departments / indices and route to the right one.
- Picture investigating *"Why is the birth rate so low?"* It isn't only a **demographic** question; **economic**, **cultural**, and **social** factors are all heavily involved.
- So the concern with this method is **crossover**: the concrete answer often lives across several *seemingly unrelated* indices, not in any single one.

### Agentic RAG

- **Model capability and prompting are decisive** here. Watch the token budget.
- The ultimate form of the "corporate structure" analogy — strongest on **quality and certainty**, at the cost of tokens and latency.

---

## Open gaps

Ideally the whole project would be **fully offline, private, and cheap.** The major gaps:

- **Privacy** — every token sent to an online LLM can potentially leak to third parties: the LLM provider, the cloud hosts.
- **Performance** — the gap between closed-source and open-source LLMs is still **huge**.
- **Uncertainty** — even with `temperature=0` and a fixed `seed`, outputs can vary because of engineering details: **batching**. Running fully offline, one batch at a time, removes this — but that's economically impractical when you need to process many batches in production.

---

## Sharpening the axe, beyond RAG

All costs in *this* project are **purely operational** — nothing else is incurred. In a
real-world project, the up-front **construction cost** is expected to **exceed** even the
discounted (present-value) **operational + maintenance cost**, and that's the point:
*sharpen the axe before chopping the tree.*

**How to sharpen the axe (one possible route):**

- Feed all your data — operational records, financial reports, emails, chats — into LLMs and distill it into dense `memories.md` / `skills.md` / MCP servers (highly confidential;needs regular maintenance).
  - Plug these in as input context to lift output quality.
- PageIndex (vectorless) improvements:
  - The project doesn't use an LLM to build the tree structure. Adding one could produce a clearer structure.

In real-estate investment (not only), a term called **framing bias**. How the statement/question is framed, influencing the thoughts, answers and following behaviors. Sometimes, it can cover real issues.<br>
With the distilled company-specific contexts, the LLM can help you reframe the problem; find the real pain point and restructure the assignments, making meaningful impacts and create values.

---

## Beyond the technical: the human problem

### First principles — ask before you start

The pain point is **never** the technical part. It's the **human** part — the implementation. Always ask, before kicking anything off:

- What are you trying to achieve?
- does the cost justifies itself? (Today's API prices are unprofitable for providers like OpenAI; expect them to rise.)
- Have you weighed the **adverse-selection / moral-hazard** risk from your own employees — the same way you would for the health / disability policies you write for policyholders?
- If the project doesn't pan out, what can you pivot the work toward?

### Driving adoption at the corporate level

1. Enterprise risk management has a name for this: **agency risk** — the interest misalignment between a company's *agents* (managers, employees) and its *principals* (the shareholders).
2. The three groups want different things from AI: **shareholders** want it to boost **revenue**, **managers** want **productivity**, and many **employees** mostly want to **spend less time on their work** (and more on chit-chat or TikTok). Employees are the **least** motivated to invest time learning it.
3. Possible **solutions**:
   - Frame the workflow change so it **convinces and reassures** employees: **this is her to help you, not to replace you.**
   - foster an **AI culture.** Recall how the media during the COVID years used framing, repetition, and persuasion to embed ideas at scale — the same mechanics can be applied here, in a **positive and ethical** way. Pair it with **KPIs**, discussions, hackathons, etc.
     - The board and managers should be the pioneers (a steering committee?).
     - **Front-line operating managers are the key.** People at the top of the hierarchy are far from daily operations; only front-line staff can **spot and fix** the loopholes AI introduces and give **timely feedback.** Bring them onto the committee.
     - **Time ≫ money** — result takes years to reflect.
   - Actively help employees find their place and give them real support — e.g., make them the human **quality gatekeeper** for AI output.
   - Enforce **shareholders and the C-suite** accountablity — give them **"skin in the game."** (though tough in reality) Employees and middle managers already know they're the ones who bear the fallout of a failed strategic decision.

**A caveat on all of this:** today's AI is still trained mostly on **human** data, not synthetic — so *"garbage in, garbage out"* still holds. If decision-makers aggressively push an agenda to replace people, **don't be surprised when employees pollute your data.** The result can be disappointing and expensive.

- Most industries already have best practices and SOPs but rarely follow them strictly, for convenience / less friction. AI **does not solve that structural problem.**
- Look at how many people AMZN and META laid off — and how many they're forced to rehire within a year (short-term).

### Emerging RIsk
- Current LLM behaves like a omni-PhD assistant for everyone, lack of complete personality, restrained/protected by injected prompts from "committing crimes" which can be jailbreak by users to do bad things.<br>
- **Before adopting, preparation to mitigate the potential risks is highly recommended.** Faking claims, symptoms, notes, voices, etc. can lead to surge of insurance claims and administrative costs.