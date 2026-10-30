import json
import os

log_directory = "../data/sessions_v2"  

crash_deltas_steps = []
crash_deltas_tokens = []

for filename in os.listdir(log_directory):
    if not filename.endswith(".json") and not filename.endswith(".jsonl"):
        continue
        
    filepath = os.path.join(log_directory, filename)
    
    crash_active = False
    crash_step_index = 0
    crash_token_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
                
            try:
                step = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # 1. Detect the Crash Trigger
            # The agent explicitly flags your assigned category as "runtime"
            if not crash_active and step.get("event") == "test_fail":
                if step.get("error_category") == "runtime":
                    crash_active = True
                    crash_step_index = step.get("step_id", 0)
                    crash_token_count = step.get("tokens_so_far", 0)
            
            # 2. Detect the Resolution
            # The crash is resolved when the agent successfully gets a test to pass
            elif crash_active and step.get("event") == "test_pass":
                resolution_step_index = step.get("step_id", 0)
                resolution_token_count = step.get("tokens_so_far", 0)
                
                # 3. Calculate Deltas
                steps_to_resolve = resolution_step_index - crash_step_index
                tokens_to_resolve = resolution_token_count - crash_token_count
                
                crash_deltas_steps.append(steps_to_resolve)
                crash_deltas_tokens.append(tokens_to_resolve)
                
                # Reset to catch any subsequent crashes in the same file
                crash_active = False

# 4. Compute Final "Crash" Averages
if crash_deltas_tokens:
    avg_tokens = sum(crash_deltas_tokens) / len(crash_deltas_tokens)
    avg_steps = sum(crash_deltas_steps) / len(crash_deltas_steps)
    
    print(f"Total Crash instances resolved: {len(crash_deltas_tokens)}")
    print(f"Average tokens since failure (X for Base): {avg_tokens:.2f}")
    print(f"Average steps to resolve: {avg_steps:.2f}")
else:
    print("No resolved runtime crashes found in the parsed logs.")