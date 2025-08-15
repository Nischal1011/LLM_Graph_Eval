import os, json, re, time
from pathlib import Path
from openai import OpenAI

# =================== CONFIG ===================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # set this in the environment
MODEL_BASE = "gpt-4o"          # Base LLM (answers from raw source text)
MODEL_KG = "gpt-4o"            # KG LLM (answers from KG only; swap to your KG-backed endpoint if needed)
MODEL_JUDGE = "gpt-4o"         # Independent judge (use GPT-5 model; adjust name if different in your org)

TEMPERATURE_GEN = 0.0          # Deterministic generation for Base & KG
TEMPERATURE_JUDGE = 0.0        # Deterministic judging
RATE_LIMIT_SLEEP = 0.2         # Tune for rate limits

SOURCE_TEXT_PATH = "data/generated_sample.md"  # single combined source text
KG_JSON_PATH = "data/knowledge_graph.json"     # single knowledge graph JSON
QUESTIONS_DIR = "data/questions"               # *.md question files by category
OUTPUT_JSONL = "judgments.jsonl"
# ==============================================

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------- IO Helpers -----------------
def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_md_questions(md_text: str):
    """Parse numbered/bulleted markdown into a clean list of questions."""
    lines = [l.strip() for l in md_text.splitlines()]
    qs = []
    for l in lines:
        if not l:
            continue
        l = re.sub(r"^\s*(\d+\.)\s*", "", l)  # remove "1. "
        l = re.sub(r"^\s*[-*]\s*", "", l)     # remove "- " or "* "
        if l:
            qs.append(l)
    return qs

def chat_once(model: str, prompt: str, temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content

# ----------------- Prompts --------------------


def base_answer_prompt(source_text: str, question: str) -> str:
    """Prompt for base LLM using source text only"""
    return f"""Answer the question using ONLY the Source Text below. Do not use external knowledge.

[Source Text]
{source_text}

[Question]
{question}

Provide a clear, well-reasoned answer based on the source text.
"""


def kg_answer_prompt(knowledge_graph: str, question: str) -> str:
    """Allows reasoning over KG but restricts factual claims to KG content"""
    return f"""Answer the question using the Knowledge Graph below. You may use reasoning and inference, but base all factual claims strictly on the information in the knowledge graph.

[Knowledge Graph]
{knowledge_graph}

[Question]
{question}

Use logical reasoning over the knowledge graph relationships, but do not introduce facts not represented in the graph. If you need to make inferences, clearly indicate your reasoning process.
"""



def judge_prompt(source_text: str, question: str, ans_base: str, ans_kg: str) -> str:
    # Judge uses ONLY the Source Text as ground truth to evaluate both answers.
    return f"""You are a helpful assistant responsible for grading two answers to a question that are provided by two different systems: a Base LLM and a Knowledge Graph. You must judge them only using the provided Source Text. Do not use any external knowledge.

Metrics
Completeness
 How fully does the answer cover all aspects and details of the question? A complete answer is thorough and covers every relevant point from the Source Text without omitting important information.

Diversity
 How varied and rich is the answer in providing different perspectives and insights? A diverse answer uses multiple relevant viewpoints, examples, or relationships from the Source Text instead of sticking to one narrow angle.

Empowerment
 How well does the answer help the reader understand and make informed judgments about the topic? An empowering answer explains reasoning clearly, surfaces implications, and grounds them in the Source Text.

Directness
 How specifically and clearly does the answer address the question? A direct answer avoids unnecessary detail or irrelevant tangents, and delivers the needed information clearly and concisely.

Your Task
Given the Source Text, Question, and two Answers, evaluate each answer on all four metrics. Provide both a numeric score (1–5) and a short reasoning for each metric for each answer. Then, determine which answer is better overall.

Note: The Base LLM answered using the source text; the KG system answered using only a structured knowledge graph (no text). Still, grade both strictly against the Source Text.

Input Format
[Source Text]
{source_text}

[Question]
{question}

[Answer 1 — Base LLM]
{ans_base}

[Answer 2 — Knowledge Graph]
{ans_kg}

Output Format
Return your evaluation as JSON in the following structure:
{{
  "answers": {{
    "1": {{
      "scores": {{
        "completeness": 1,
        "diversity": 1,
        "empowerment": 1,
        "directness": 1
      }},
      "reasoning": {{
        "completeness": "Explain how well Answer 1 covers all parts of the question based on the Source Text.",
        "diversity": "Explain the range of perspectives, examples, or relationships used in Answer 1.",
        "empowerment": "Explain how Answer 1 helps the reader understand and make judgments grounded in the Source Text.",
        "directness": "Explain how clear and concise Answer 1 is in addressing the question."
      }}
    }},
    "2": {{
      "scores": {{
        "completeness": 1,
        "diversity": 1,
        "empowerment": 1,
        "directness": 1
      }},
      "reasoning": {{
        "completeness": "Explain how well Answer 2 covers all parts of the question based on the Source Text.",
        "diversity": "Explain the range of perspectives, examples, or relationships used in Answer 2.",
        "empowerment": "Explain how Answer 2 helps the reader understand and make judgments grounded in the Source Text.",
        "directness": "Explain how clear and concise Answer 2 is in addressing the question."
      }}
    }}
  }},
  "winner": 1,
  "winner_label": "base",   // "base" or "kg"
  "winner_reasoning": "Short, specific rationale for why the chosen answer wins.",
  "overall_reasoning": "Explain why the winning answer is better overall, citing strengths and weaknesses across the four metrics."
}}

Scoring Guidelines (1–5 per metric)
5 = Excellent: Fully meets the metric’s definition with strong evidence from the Source Text.
4 = Good: Meets most requirements with minor gaps.
3 = Fair: Addresses the metric partially but misses notable elements.
2 = Poor: Only slightly meets the metric’s intent.
1 = Very Poor: Fails to meet the metric’s intent.
"""

# ----------------- Main Loop ------------------
def main():
    # 1) Load single combined source text
    source_text = load_file(SOURCE_TEXT_PATH)

    # 2) Load single knowledge graph JSON as string
    kg_json = load_json(KG_JSON_PATH)
    kg_json_str = json.dumps(kg_json, ensure_ascii=False)

    # 3) Load questions from .md files by category
    qfiles = sorted(Path(QUESTIONS_DIR).glob("*.md"))
    items = []
    for qf in qfiles:
        category = qf.stem  # e.g., "sense_making"
        md = load_file(str(qf))
        qs = parse_md_questions(md)
        for q in qs:
            items.append({"category": category, "question": q})

    # 4) Iterate questions: Base answer (text), KG answer (KG only), Judge (uses text)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            category = item["category"]
            question = item["question"]

            # Base: from Source Text
            prompt_base = base_answer_prompt(source_text, question)
            ans_base = chat_once(MODEL_BASE, prompt_base, TEMPERATURE_GEN)
            time.sleep(RATE_LIMIT_SLEEP)

            # KG: from KG only
            prompt_kg = kg_answer_prompt(kg_json_str, question)
            ans_kg = chat_once(MODEL_KG, prompt_kg, TEMPERATURE_GEN)
            time.sleep(RATE_LIMIT_SLEEP)

            # Judge (uses Source Text as ground truth)
            jprompt = judge_prompt(source_text, question, ans_base, ans_kg)
            judge_raw = chat_once(MODEL_JUDGE, jprompt, TEMPERATURE_JUDGE)
            time.sleep(RATE_LIMIT_SLEEP)

            # Parse judge JSON (fall back to raw if needed)
            try:
                judge_json = json.loads(judge_raw)
            except json.JSONDecodeError:
                judge_json = {"_parse_error": True, "raw": judge_raw}

            # Derive selected answer + reasoning
            winner = judge_json.get("winner")
            winner_label = judge_json.get("winner_label")
            if winner_label not in {"base", "kg"}:
                # Fallback from numeric winner if label missing
                winner_label = "base" if winner == 1 else ("kg" if winner == 2 else "tie")

            if winner_label == "base":
                selected_text = ans_base
            elif winner_label == "kg":
                selected_text = ans_kg
            else:
                selected_text = None  # tie or parse error

            selected_reasoning = judge_json.get("winner_reasoning") or judge_json.get("overall_reasoning")

            record = {
                "idx": i,
                "category": category,
                "question": question,
                "answer_base": ans_base,
                "answer_kg": ans_kg,
                "judge": judge_json,  # full judge JSON
                "selected": {         # convenient summary for your logs
                    "system": winner_label,            # "base" | "kg" | "tie"
                    "answer_text": selected_text,      # the chosen answer's text (if any)
                    "reasoning": selected_reasoning    # why it won, from judge
                }
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(items)}] {category} — selected: {record['selected']['system']}")

if __name__ == "__main__":
    main()
