from anthropic import Anthropic
from dotenv import load_dotenv

import os
import subprocess

load_dotenv(override=True)

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)
MODEL = os.environ["MODEL_ID"]
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ── Tool definition: just bash ────────────────────────────
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
},{
    "name": "write_file",
    "description": "Write content to a file.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, 
        "required": ["path", "content"],
    },
},{
    "name": "read_file",
    "description": "Read file contents.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, 
        "required": ["path"],
    },
},{
    "name": "delete_file",
    "description": "Delete a file.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}}, 
        "required": ["path"],
    },
}]


# ── Tool execution ────────────────────────────────────────
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    try:
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {abs_path}"
    except (OSError, IOError) as e:
        return f"Error: {e}"


def read_file(path: str, limit: int = None) -> str:
    try:
        abs_path = os.path.abspath(path)
        with open(abs_path, "r") as f:
            if limit:
                lines = f.readlines()[:limit]
                content = "".join(lines)
            else:
                content = f.read()
        return content if content else "(empty file)"
    except (OSError, IOError) as e:
        return f"Error: {e}"


def delete_file(path: str) -> str:
    try:
        abs_path = os.path.abspath(path)
        os.remove(abs_path)
        return f"Successfully deleted {abs_path}"
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except (OSError, IOError) as e:
        return f"Error: {e}"

TOOL_HANDLERS = {
    "bash": lambda inp: run_bash(inp["command"]),
    "write_file": lambda inp: write_file(inp["path"], inp["content"]),
    "read_file": lambda inp: read_file(inp["path"], inp.get("limit")),
    "delete_file": lambda inp: delete_file(inp["path"]),
}


def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, 
            system=SYSTEM, 
            tools=TOOLS,
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
                print(f"\033[33m[{block.name}] {block.input}\033[0m")
                handler = TOOL_HANDLERS.get(block.name)
                if handler:
                    output = handler(block.input)
                else:
                    output = f"Error: Unknown tool '{block.name}'"
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