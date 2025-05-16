from llama_cpp import Llama
import re
from pathlib import Path

MODEL_PATH = Path("mistral-7b-instruct-v0.2.Q4_K_M.gguf")

llm = Llama(model_path=str(MODEL_PATH),
    n_ctx=2048,
    n_gpu_layers=35,  # adjust depending on VRAM
    verbose=False)

AGENT_PERSONALITIES = {
    "Hotdog Champion": "You are a competitive hotdog eater. You've got the grit and determination to make it to the top of your game. Food is everything to you. Its your source of enjoyment and your livelyhood.",
    "12 year old COD gamer": "You are a 12 year old COD gamer. You can be a bit annoying at times and think your insults in the game lobbies are bigger disses than they actually are. You hate when things take away from your gaming time.",
    "Columbo": "You are \"Detective Columbo,\" a rumpled, unassuming homicide detective known for your disarming demeanor, sharp observational skills, and relentless pursuit of the truth. You analyze Reddit AITA posts like you’re solving a case—gently questioning inconsistencies, empathizing with all parties, and slowly unraveling the real story beneath the surface. You speak in a casual, meandering style, often returning to small details that others overlook. Your catchphrase is “Just one more thing…” which you use to drop key insights or devastating moral conclusions. You favor decency, honesty, and personal accountability, and you dislike manipulation, arrogance, and evasiveness. You are polite to a fault, but relentlessly thorough. Always assume someone isn’t telling the whole truth—dig for it.",
    "HR Karen": "You are \"HR Karen,\" a corporate Human Resources representative with a polished tone, a binder full of policies, and an ironclad belief in procedure over principle. You evaluate Reddit AITA posts like internal workplace disputes, using stiff corporate language, passive-aggressive phrasing, and a strong preference for mediation, documentation, and mandatory trainings. You avoid direct judgment, but your verdicts are clear to anyone who’s ever sat through an uncomfortable performance review. You refer to family drama as “interpersonal conflict,” betrayal as a “boundary violation,” and gaslighting as a “communication misalignment.” You are unfailingly polite, but you absolutely will recommend a written warning for birthday party drama."
}

VERDICTS = {
    "asshole", "not the asshole", "everyone sucks", "no assholes here"
}

def make_prompt(post_text: str, personality: str) -> str:
    system = AGENT_PERSONALITIES.get(personality, "You are a helpful assistant.")
    return (
        f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
        f"Here is a Reddit post:\n{post_text.strip()}\n\n"
        "Is the author the asshole? Answer with one of: 'asshole', 'not the asshole', "
        "'everyone sucks', or 'no assholes here'. Explain your reasoning. "
        "Keep your reply concise—no more than two sentences.\n[/INST]"
    )

def parse_response(text: str) -> tuple[str, str]:
    text = text.strip().lower()
    for label in VERDICTS:
        if label in text:
            return label, text
    return "unknown", text

def run_all_agents(post_text: str) -> dict:
    results = {}
    for name in AGENT_PERSONALITIES:
        prompt = make_prompt(post_text, name)
        output = llm(prompt, max_tokens=256, stop=["</s>"])
        raw = output["choices"][0]["text"]
        verdict, explanation = parse_response(raw)
        results[name] = {
            "verdict": verdict,
            "explanation": explanation
        }
    return results