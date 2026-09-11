from src.promptConstructor import PromptConfig, Template, TemplateRegistry, ContextTruncator, TokenCounter, MessageFormatter

class PromptBuilder:
    def __init__(self, config: PromptConfig, template_registry: TemplateRegistry = None):
        self.config = config
        self.template_registry = template_registry if template_registry is not None else TemplateRegistry()
        self.tokenizer = TokenCounter(config)
        self.truncator = ContextTruncator(config)
        self.formatter = MessageFormatter(config)

    def build(self, 
              query: str,
              docs: list,
              history: list = None,
              scene: str = "qa") -> list:
        template = self.template_registry.get(scene)
        system_prompt = self._build_system_prompt(template)
        formatted_history = self._format_history(history) if history else ""
        context = self._organize_context(docs)
        truncated_context = self._truncate_context(context)
        user_message = self._build_user_message(template, query, truncated_context, formatted_history)
        return self.formatter.format(system_prompt, user_message, history)

    def _organize_context(self, docs: list) -> str:
        return

    def _truncate_context(self, context: str) -> str:
        return

    def _format_history(self, history: list) -> str:
        return

    def _build_system_prompt(self, template: Template) -> str:
        return

    def _build_user_message(self, template: Template, query: str, context: str, history: list) -> str:
        return
