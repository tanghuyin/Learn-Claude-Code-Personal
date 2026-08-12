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

# ── Permission system ─────────────────────────────────────

WORKSPACE = os.getcwd()


def is_path_in_workspace(path: str) -> bool:
    """Check if a resolved path is within the workspace directory."""
    abs_path = os.path.abspath(path)
    return abs_path.startswith(os.path.abspath(WORKSPACE) + os.sep) or abs_path == os.path.abspath(WORKSPACE)


def check_file_permissions(args: dict) -> bool:
    """File tools: path must be inside workspace."""
    path = args.get("path", "")
    if not is_path_in_workspace(path):
        print(f"\033[31m[DENIED] path '{path}' is outside workspace\033[0m")
        return False
    return True


def ask_user_decision(message: str, command: str) -> bool:
    """Prompt user to allow or deny a risky operation."""
    print(f"\033[33m[CAUTION] {message}\033[0m")
    print(f"  Command: {command}")
    resp = input("  Allow? (y/n): ").strip().lower()
    if resp != "y":
        print(f"\033[31m[DENIED] User rejected command\033[0m")
        return False
    return True


def check_bash_permissions(args: dict) -> bool:
    """Bash: deny dangerous commands, prompt for risky ones."""
    command = args.get("command", "")

    # Hard deny
    deny_patterns = [
        "rm -rf /", "rm -rf /*", "sudo rm -rf", "mkfs",
        "dd if=", "> /dev/sda", "shutdown", "reboot",
        ":(){ :|:& };:",  # fork bomb
    ]
    for pattern in deny_patterns:
        if pattern in command:
            print(f"\033[31m[DENIED] blocked pattern '{pattern}'\033[0m")
            return False

    # Caution — ask user
    caution_patterns = [
        "sudo", "rm -rf", "git push --force", "git reset --hard",
        "curl | sh", "curl | bash", "wget | sh",
        "chmod 777", "npm publish", "pip install",
    ]
    for pattern in caution_patterns:
        if pattern in command:
            return ask_user_decision(f"Command contains '{pattern}'", command)

    return True


# Maps tool name -> permission checker function
PERMISSION_CHECKS = {
    "bash": check_bash_permissions,
    "write_file": check_file_permissions,
    "read_file": check_file_permissions,
    "delete_file": check_file_permissions,
}


def check_permissions(block) -> bool:
    """Returns True if the tool use is allowed, False if denied."""
    checker = PERMISSION_CHECKS.get(block.name)
    if checker:
        return checker(block.input)
    return True



def agent_loop(messages: list, max_denials: int = 3):
    denial_count = 0

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
                # check permission before using the tool
                if not check_permissions(block):
                    denial_count += 1
                    if denial_count >= max_denials:
                        print(f"\033[31m[ABORT] Too many denied attempts ({max_denials}), stopping agent.\033[0m")
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": "Permission denied. The user has repeatedly rejected this action. Do not retry — explain what you were trying to do instead."})
                        messages.append({"role": "user", "content": results})
                        return
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "Permission denied. Do not retry this action without user approval."})
                    continue

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