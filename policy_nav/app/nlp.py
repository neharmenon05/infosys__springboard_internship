import subprocess

def generate_summary(text: str) -> str:
    """Generate AI summary using local LLaMA via Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "run", "llama2", "--text", text],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"
