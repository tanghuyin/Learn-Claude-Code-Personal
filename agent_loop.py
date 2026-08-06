from anthropic import Anthropic
from dotenv import load_dotenv

import os

load_dotenv(override=True)

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)
MODEL = os.environ["MODEL_ID"]
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, 
            system=SYSTEM, 
            messages=messages,
            max_tokens=1024,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})
        # If the model didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            return

        # Execute each tool call, collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                print(output[:200])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        # Feed tool results back, loop continues
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("Agent Loop")
    print("Type the prompt, enter to send. q to quit. \n")

    history = []
    while True:
        try:
            query = input()
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "quit"):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print(history)
        print()