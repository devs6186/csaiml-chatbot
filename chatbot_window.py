import llm_integration
from pygments import highlight, lexers, formatters
import sys
try:
    import readline  # Works on Unix/Linux
except ImportError:
    import pyreadline3 as readline  # Fallback for Windows

def colored_print(text: str, color: str = None, end: str = '\n'):
    """Print colored text in terminal"""
    colors = {
        'user': '\033[94m',    # Blue
        'bot': '\033[92m',     # Green
        'system': '\033[90m',  # Gray
        'error': '\033[91m',   # Red
        'reset': '\033[0m'
    }
    if color in colors:
        print(f"{colors[color]}{text}{colors['reset']}", end=end)
    else:
        print(text, end=end)

def format_response(response: str) -> str:
    """Highlight code blocks using pygments if present in the LLM response"""
    if '```' in response:
        try:
            parts = response.split('```')
            formatted = parts[0]
            for i in range(1, len(parts), 2):
                lang = "python" if parts[i].startswith("python") else "text"
                code = parts[i].replace("python", "", 1) if lang == "python" else parts[i]
                highlighted = highlight(code.strip(), lexers.get_lexer_by_name(lang), formatters.TerminalFormatter())
                formatted += f"\n{highlighted}\n" + parts[i+1] if i+1 < len(parts) else ""
            return formatted
        except:
            return response
    return response

def terminal_chat():
    print("\n" + "="*55)
    colored_print("📘  Course Document Q&A System", 'system')
    colored_print("Type 'exit' or 'quit' to end the session\n", 'system')
    print("="*55 + "\n")

    while True:
        try:
            colored_print("You: ", 'user', end='')
            query = input().strip()

            if query.lower() in ('exit', 'quit'):
                colored_print("\n👋 Session ended. Goodbye!\n", 'system')
                break

            if not query:
                continue

            response = llm_integration.search_and_generate(query)
            colored_print("\nBot:", 'bot')
            print(format_response(response) + "\n")

        except KeyboardInterrupt:
            colored_print("\n🛑 Interrupted. Exiting...\n", 'system')
            break
        except Exception as e:
            colored_print(f"\nError: {str(e)}\n", 'error')

if __name__ == "__main__":
    terminal_chat()