from dataclasses import dataclass

@dataclass
class PromptConfig:
    '''
    A dataclass to hold the configuration for prompt construction.
    '''

    # role configuration
    role: str = ""
    role_description: str = ""

    # Context configuration
    max_context_tokens: int = 4000
    max_docs: int = 5
    context_separator: str = "\n\n---\n\n"
    include_metadata: bool = True

    # hitorical context configuration
    max_history_rounds: int = 3
    include_history: bool = True

    # output configuration
    output_format: str = "markdown"   # markdown | json | plain
    language: str = "auto"            # auto | zh | en
    
    # Generative configuration
    temperature: float = 0.1
    max_tokens: int = 1500


@dataclass
class Template:
    name: str
    system_extra: str = ""
    user_extra: str = ""
    temperature: float = None

class TemplateRegistry:
    """管理不同场景的模板"""
    
    def __init__(self):
        self._templates = {
            "qa": Template(
                name="qa",
                user_extra="## Requirements\nAnswer directly. Cite sources as [1], [2]."
            ),
            "summarize": Template(
                name="summarize",
                system_extra="## Task\nSummarize the key points concisely.",
                user_extra="## Requirements\nProvide a structured summary with bullet points.",
                temperature=0.3
            ),
            "compare": Template(
                name="compare",
                system_extra="## Task\nCompare and contrast the given information.",
                user_extra="## Requirements\nUse a comparison table."
            ),
            "howto": Template(
                name="howto",
                system_extra="## Task\nProvide step-by-step instructions.",
                user_extra="## Requirements\nNumber each step."
            ),
        }
    
    def get(self, scene: str) -> Template:
        return self._templates.get(scene, self._templates["qa"])
    
    def register(self, template: Template):
        self._templates[template.name] = template


class ContextTruncator:
    """context truncation strategies for prompt construction"""
    
    def __init__(self, config: PromptConfig):
        self.config = config
    
    def truncate(self, context: str, max_tokens: int) -> str:
        if len(context) <= max_tokens:
            return context
        
        return self._head_tail_truncate(context, max_tokens)
    
    def _head_tail_truncate(self, text: str, max_tokens: int) -> str:
        head = max_tokens // 2
        tail = max_tokens - head
        return text[:head] + "\n...[truncated]...\n" + text[-tail:]


class TokenCounter:
    
    def __init__(self, model: str = "Qwen-7B"):
        self.model = model

    
    def count(self, text: str) -> int:
        return


class MessageFormatter:
    
    def __init__(self, config: PromptConfig):
        self.config = config
    
    def format(self, system_prompt: str, user_message: str,
               history: list = None) -> list:
        messages = [{"role": "system", "content": system_prompt}]
        
        return messages