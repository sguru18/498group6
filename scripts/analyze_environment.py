import json
import os

log_directory = "../data/sessions_v2"

env_deltas_steps = []
env_deltas_tokens = []

for filename in os.listdir(log_directory):
    if not filename.endswith(".json") and not filename.endswith(".jsonl"):
        continue

    filepath = os.path.join(log_directory, filename)

    env_active = False
    env_step_index = 0
    env_token_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                step = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 1. Detect the Environment Trigger
            if not env_active and step.get("event") == "test_fail":
                if step.get("error_category") == "environment":
                    env_active = True
                    env_step_index = step.get("step_id", 0)
                    env_token_count = step.get("tokens_so_far", 0)

            # 2. Detect the Resolution
            elif env_active and step.get("event") == "test_pass":
                resolution_step_index = step.get("step_id", 0)
                resolution_token_count = step.get("tokens_so_far", 0)

                steps_to_resolve = resolution_step_index - env_step_index
                tokens_to_resolve = resolution_token_count - env_token_count

                env_deltas_steps.append(steps_to_resolve)
                env_deltas_tokens.append(tokens_to_resolve)

                env_active = False

if env_deltas_tokens:
    avg_tokens = sum(env_deltas_tokens) / len(env_deltas_tokens)
    avg_steps = sum(env_deltas_steps) / len(env_deltas_steps)

    print(f"Total Environment instances resolved: {len(env_deltas_tokens)}")
    print(f"Average tokens since failure (X): {avg_tokens:.2f}")
    print(f"Average steps to resolve: {avg_steps:.2f}")
else:
    print("No resolved environment failures found in the parsed logs.")