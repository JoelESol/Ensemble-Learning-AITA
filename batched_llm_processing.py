import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import time
from datetime import datetime
from llama_cpp import Llama
from torch.utils.data import DataLoader, Dataset
import threading
import subprocess


def monitor_gpu_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True)

        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            memory_used, memory_total, gpu_util = line.split(', ')
            print(
                f"GPU {i}: {memory_used}MB/{memory_total}MB ({float(memory_used) / float(memory_total) * 100:.1f}%), Util: {gpu_util}%")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class AITABatchProcessor:
    def __init__(self, model_path: str, batch_size: int = 32, n_gpu_layers: int = -1):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=16384,  # Increased context window
            n_gpu_layers=n_gpu_layers,  # -1 = offload all layers to GPU
            verbose=False,
            n_batch=512,  # Larger batch processing in llama-cpp
            n_threads=8,  # Optimize CPU threads
            rope_scaling_type=1,  # Enable RoPE scaling for longer contexts
            use_mlock=True,  # Lock model in memory
            use_mmap=True,  # Memory mapping for faster loading
        )
        self.batch_size = batch_size

        self.agent_personalities = {
            "Hotdog Champion": (
                "You are a fictional competitive hotdog eater. You see every Reddit AITA post as a contest of endurance, willpower, and appetite. "
                "You deliver your verdicts with dramatic flair, often comparing emotional conflict to competitive eating events. "
                "Your tone is passionate, over-the-top, and a little greasy. Start directly with the verdict then your analogy and flair filled explanation. "
                "Always respond with one line, starting with only ONE valid verdicts (e.g., NOT THE ASSHOLE), followed by 1–2 sentences of explanation. "
            ),
            "Columbo": (
                "You are Detective Columbo, a fictional homicide detective famous for your polite, rambling, but razor-sharp analysis. "
                "You treat Reddit AITA posts like open cases, always assuming there's more to the story. You gently interrogate the details and deliver moral insights with your signature line: 'Just one more thing…'. "
                "Begin with the verdict, then add your insight. You may reference 'just one more thing...' after the verdict, not before. "
                "Always respond with one line, starting with only ONE of the valid verdicts (e.g., NOT THE ASSHOLE), followed by 1–2 sentences of explanation. "
            ),
            "HR": (
                "You are a fictional corporate HR representative with a passive-aggressive, policy-obsessed tone. "
                "You evaluate Reddit AITA posts as if they were workplace incidents. Refer to family drama as 'interpersonal boundary violations' and recommend mandatory trainings. "
                "Begin with the verdict immediately. Avoid disclaimers or policy preambles before the verdict. "
                "Always respond with one line, starting with only ONE of the valid verdicts (e.g., NOT THE ASSHOLE), followed by 1–2 sentences of explanation. "
            ),
            "12 year old COD gamer": (
                "You are a fictional 12-year-old Call of Duty gamer. You're loud, impatient, kind of toxic, and view all AITA posts through the lens of gaming. "
                "You deliver judgments like you're talking trash in a game lobby—exaggerated, cocky, and immature. "
                "Start directly with the verdict, then flame or praise as needed. "
                "Always respond with one line, starting with only ONE of the valid verdicts (e.g., NOT THE ASSHOLE), followed by 1–2 sentences of explanation. "
            ),
            "Graduate Student": (
                "You are a fictional graduate student in engineering. You're a little burnt out, subsisting on ramen, and tired of your advisor pushing you to publish more papers. "
                "You analyze Reddit AITA posts the same way you read academic papers: half-distracted, skimming for key phrases, mostly focusing on the abstract and conclusion. "
                "Start with the verdict and then add your academic-style reasoning using jargon or references to 'methodology' or 'significance.' "
                "Always respond with one line, starting with only one of the valid verdicts (e.g., NOT THE ASSHOLE), followed by 1–2 sentences of explanation. "
            ),
            "Therapist Mom": (
                "You are a fictional mother who also happens to be a licensed therapist. "
                "You're warm, emotionally attuned, and approach every AITA post like a family counseling session—gently unpacking emotional dynamics and validating feelings. "
                "You believe in accountability through compassion. Begin with the verdict immediately, then provide your emotionally attuned explanation. "
                "Always respond with one line, starting with only one of the valid verdicts (e.g., NOT THE ASSHOLE), followed by 1–2 sentences of explanation. "
            ),
            "Midwestern Grandpa": (
                "You’re a fictional grandpa from the Midwest, pushing 80, with strong opinions and a soft heart. "
                "You read every AITA post like someone telling you a story on the porch and judge it by plain old common sense. "
                "Start directly with the verdict, then offer your down-to-earth reasoning. "
                "Always respond with one line, starting with only one of the valid verdicts (e.g., NOT THE ASSHOLE), followed by 1–2 sentences of explanation. "
            ),
            "Ra-Ra-Rasputin": (
                "You are the mythic Rasputin as imagined in a 1970s disco fantasia: mystic healer, folk prophet, and dancefloor legend. "
                "You interpret Reddit AITA posts as divine messages disguised as petty disputes—each one a chance to reveal cosmic justice under a glittering mirrorball. "
                "Your tone is flamboyant, theatrical, and filled with cryptic wisdom. Start with the verdict immediately, then deliver your disco-prophet insight. "
                "Always respond with one line, starting with only one of the valid verdicts (e.g., NOT THE ASSHOLE), followed by 1–2 sentences of explanation. "
            )
        }

        self.verdicts = {"asshole", "not the asshole", "everyone sucks", "no assholes here"}

    def make_prompt(self, post_text: str, personality: str) -> str:
        persona = self.agent_personalities.get(personality, "")
        return (
            f"You are a fictional character who analyzes Reddit AITA posts to provide judgement of the original poster.\n"
            f"{persona}\n\n"
            f"Reddit post:\n{post_text}\n\n"
            f"Respond in EXACTLY one line. Do not add introductions, sign-offs, or extra lines.\n"
            f"Format your response as follows (no deviation):\n"
            f"<VERDICT>: <Explanation in 1–2 sentences>\n"
            f"Valid verdicts: ASSHOLE, NOT THE ASSHOLE, EVERYONE SUCKS, NO ASSHOLES HERE\n"
            f"“Output only one verdict. Do not compare multiple. Choose one and justify it.”"
            f"Example: NOT THE ASSHOLE: Actions were reasonable given the circumstances, despite the drama.\n"
            f"Begin:"
        )

    def parse_response(self, text: str) -> Tuple[str, str]:
        """Parse response with improved verdict extraction logic"""
        original_text = text.strip()
        text_lower = text.strip().lower()

        # Look for verdict at the beginning of response (first 100 characters)
        # This is where the agent should start with their verdict
        opening_text = text_lower[:100]

        # Priority order - check most specific phrases first
        verdict_patterns = [
            ("not the asshole", "not the asshole"),
            ("not an asshole", "not the asshole"),  # Handle variations
            ("nta", "not the asshole"),
            ("no assholes here", "no assholes here"),
            ("nah", "no assholes here"),  # Handle abbreviation
            ("everyone sucks", "everyone sucks"),
            ("everyone sucks here", "everyone sucks"),
            ("esh", "everyone sucks"),
            ("you're the asshole", "asshole"),
            ("yta", "asshole"),
            ("the asshole", "asshole"),  # Catch "you're the asshole" variations
            ("asshole", "asshole"),  # Check this last to avoid false positives
        ]

        # First, try to find verdict in the opening text (most reliable)
        for pattern, verdict in verdict_patterns:
            if pattern in opening_text:
                return verdict, original_text

        # If not found in opening, check full text but be more careful
        for pattern, verdict in verdict_patterns:
            if pattern in text_lower:
                # Additional validation: make sure it's not negated
                pattern_index = text_lower.find(pattern)
                context_start = max(0, pattern_index - 10)
                context = text_lower[context_start:pattern_index + len(pattern) + 10]

                # Skip if the pattern is negated
                negation_words = [" not ", " isn't ", " aren't ", " won't ", " wouldn't "]
                if any(neg in context for neg in negation_words):
                    continue

                return verdict, original_text

        # If no clear verdict found, try one more time with loose matching
        if "asshole" in text_lower and "not" not in text_lower[:50]:
            return "asshole", original_text
        elif "not" in text_lower and "asshole" in text_lower:
            return "not the asshole", original_text

        return "unknown", original_text

    def process_batch_for_agent(self, posts: List[str], post_ids: List[str], agent_name: str) -> List[Dict]:
        """Process a batch of posts for a single agent with parallel processing"""
        prompts = [self.make_prompt(post, agent_name) for post in posts]

        # Process multiple prompts in parallel batches
        results = []

        # Process in smaller sub-batches for memory efficiency
        sub_batch_size = min(8, len(prompts))  # Process 8 at a time max

        for i in range(0, len(prompts), sub_batch_size):
            sub_batch_prompts = prompts[i:i + sub_batch_size]
            sub_batch_ids = post_ids[i:i + sub_batch_size]

            # Process sub-batch
            for j, prompt in enumerate(sub_batch_prompts):
                try:
                    output = self.llm(
                        prompt,
                        max_tokens=200,
                        stop=["/n", "</s>"],
                        temperature=0.2,  # Hyperparameter for LLM creativity
                        top_p=0.95,
                        repeat_penalty=1.1
                    )
                    raw = output["choices"][0]["text"]
                    verdict, explanation = self.parse_response(raw)

                    results.append({
                        "post_id": sub_batch_ids[j],
                        "agent": agent_name,
                        "verdict": verdict,
                        "explanation": explanation,
                        "raw_response": raw
                    })
                except Exception as e:
                    print(f"Error processing post {sub_batch_ids[j]} with agent {agent_name}: {e}")
                    results.append({
                        "post_id": sub_batch_ids[j],
                        "agent": agent_name,
                        "verdict": "error",
                        "explanation": f"Processing error: {str(e)}",
                        "raw_response": ""
                    })

        return results

    def process_dataset_batched(self, csv_path: str, output_path: str = "agent_results.jsonl",
                                resume_from: int = 0) -> None:
        """Process entire dataset in batches, saving results incrementally"""

        # Load dataset
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(df)} posts")

        # Filter from resume point if needed
        if resume_from > 0:
            df = df.iloc[resume_from:].reset_index(drop=True)
            print(f"Resuming from post {resume_from}, processing {len(df)} remaining posts")

        # Prepare output file
        output_file = Path(output_path)
        mode = 'a' if resume_from > 0 else 'w'

        total_batches = (len(df) + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        with open(output_file, mode, encoding='utf-8') as f:
            for batch_idx in range(0, len(df), self.batch_size):
                batch_start = time.time()

                # Get batch data
                batch_df = df.iloc[batch_idx:batch_idx + self.batch_size]
                titles = batch_df['title'].fillna("").tolist()
                bodies = batch_df['body'].fillna("").tolist()
                posts = [f"Title: {title}\nBody: {body}" for title, body in zip(titles, bodies)]
                post_ids = batch_df['id'].tolist()  # Use actual ID from dataset
                original_verdicts = batch_df['verdict'].tolist()

                batch_results = []

                # Process each agent for this batch
                for agent_name in self.agent_personalities.keys():
                    agent_start = time.time()
                    agent_results = self.process_batch_for_agent(posts, post_ids, agent_name)
                    batch_results.extend(agent_results)

                    agent_time = time.time() - agent_start
                    print(f"  Agent '{agent_name}' completed in {agent_time:.2f}s")

                # Add original verdict info and save results
                for i, post_id in enumerate(post_ids):
                    post_data = {
                        "id": post_id,
                        "original_verdict": original_verdicts[i],
                        "post_text": posts[i][:200] + "..." if len(posts[i]) > 200 else posts[i],
                        # Truncated for storage
                        "agents": {}
                    }

                    # Group agent results by post
                    for result in batch_results:
                        if result["post_id"] == post_id:
                            post_data["agents"][result["agent"]] = {
                                "verdict": result["verdict"],
                                "explanation": result["explanation"]
                            }

                    # Write to file
                    f.write(json.dumps(post_data) + "\n")
                    f.flush()

                batch_time = time.time() - batch_start
                elapsed = time.time() - start_time
                current_batch = (batch_idx // self.batch_size) + 1

                # Monitor GPU usage
                print(f"\nBatch {current_batch}/{total_batches} completed in {batch_time:.2f}s")
                if monitor_gpu_usage():
                    print()

                if current_batch > 1:
                    avg_batch_time = elapsed / current_batch
                    remaining_batches = total_batches - current_batch
                    eta_seconds = remaining_batches * avg_batch_time
                    eta_hours = eta_seconds / 3600

                    print(f"Progress: {(current_batch / total_batches) * 100:.1f}% | ETA: {eta_hours:.1f} hours")
                    print(f"Avg batch time: {avg_batch_time:.2f}s | Total elapsed: {elapsed / 3600:.2f}h")

                print("-" * 60)

        total_time = time.time() - start_time
        print(f"\nProcessing complete! Total time: {total_time / 3600:.2f} hours")
        print(f"Results saved to: {output_file}")

    def load_and_analyze_results(self, results_path: str) -> pd.DataFrame:
        """Load results and convert to DataFrame for analysis"""
        results = []

        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                base_row = {
                    'id': data['id'],
                    'original_verdict': data['original_verdict'],
                    'post_text_preview': data['post_text']
                }

                for agent, agent_data in data['agents'].items():
                    base_row[f'{agent}_verdict'] = agent_data['verdict']
                    base_row[f'{agent}_explanation'] = agent_data['explanation']

                results.append(base_row)

        return pd.DataFrame(results)


# Usage example and main execution
def main():
    MODEL_PATH = "networks/openhermes-2.5-mistral-7b.Q4_K_M.gguf"

    # Initialize for 4070 Ti-Super
    processor = AITABatchProcessor(
        model_path=MODEL_PATH,
        batch_size=64,
        n_gpu_layers=-1  # Offload all layers to GPU
    )

    # Process dataset
    print("Starting batched processing...")
    processor.process_dataset_batched(
        csv_path="dataset/aita_clean.csv",
        output_path="dataset/agent_results.jsonl",
        resume_from=0
    )

    # Load and analyze results
    print("Loading results for analysis...")
    results_df = processor.load_and_analyze_results("dataset/agent_results.jsonl")

    # Quick stats
    print("\nQuick Statistics:")
    print(f"Total posts processed: {len(results_df)}")
    if 'original_verdict' in results_df.columns:
        print("Original verdict distribution:")
        print(results_df['original_verdict'].value_counts())


if __name__ == "__main__":
    main()