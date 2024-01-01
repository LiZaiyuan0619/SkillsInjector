import importlib
import inspect
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional
import yaml
from tau2.agent.llm_agent import LLMAgent
from tau2.environment.tool import Tool, as_tool
from tau2.environment.toolkit import ToolKitBase
TEMP_DIR = Path(__file__).parent
if str(TEMP_DIR) not in sys.path:
    sys.path.insert(0, str(TEMP_DIR))

class SkillInjector:

    def __init__(self):
        self._skill_funcs: Dict[str, Callable] = {}
        self._skill_tools: Dict[str, Tool] = {}
        self._skill_metadata: Dict[str, dict] = {}
        self._prompt_guidance_parts: List[str] = []
        self._applied = False
        self._original_agent_init = None
        self._original_system_prompt = None
        self._original_use_tool = None
        self._original_has_tool = None

    def add_skill(self, func: Callable, metadata: Optional[dict]=None) -> 'SkillInjector':
        name = func.__name__
        self._skill_funcs[name] = func
        self._skill_tools[name] = as_tool(func)
        if metadata:
            self._skill_metadata[name] = metadata
        return self

    def add_prompt_guidance(self, text: str) -> 'SkillInjector':
        self._prompt_guidance_parts.append(text)
        return self

    def load_skill_module(self, module_path: str, overrides: Optional[dict]=None) -> 'SkillInjector':
        module_path = module_path.replace('/', '.').replace('\\', '.')
        if not module_path.startswith('skills.'):
            module_path = 'skills.' + module_path
        module = importlib.import_module(module_path)
        skill_name = getattr(module, 'SKILL_NAME')
        skill_func = getattr(module, skill_name)
        if overrides:
            if 'docstring' in overrides:
                skill_func.__doc__ = overrides['docstring']
            if 'body_content' in overrides:
                new_body = overrides['body_content']
                orig_sig = inspect.signature(skill_func)
                orig_name = skill_func.__name__
                orig_doc = skill_func.__doc__

                def _make_wrapper(body_text, sig):
                    params = list(sig.parameters.keys())

                    def wrapper(**kwargs):
                        try:
                            return body_text.format(**kwargs)
                        except (KeyError, IndexError):
                            return body_text
                    wrapper.__name__ = orig_name
                    wrapper.__doc__ = orig_doc
                    wrapper.__signature__ = sig
                    return wrapper
                skill_func = _make_wrapper(new_body, orig_sig)
        metadata = {'domain': getattr(module, 'SKILL_DOMAIN', 'unknown'), 'description': getattr(module, 'SKILL_DESCRIPTION', ''), 'module': module_path}
        self.add_skill(skill_func, metadata=metadata)
        guidance = (overrides or {}).get('prompt_guidance') or getattr(module, 'PROMPT_GUIDANCE', None)
        if guidance:
            self.add_prompt_guidance(guidance)
        return self

    @classmethod
    def from_config(cls, config_path: str) -> 'SkillInjector':
        with open(config_path) as f:
            config = yaml.safe_load(f)
        injector = cls()
        for skill_path in config.get('skills', []):
            injector.load_skill_module(skill_path)
        return injector

    def apply(self):
        if self._applied:
            raise RuntimeError('SkillInjector already applied. Call restore() first.')
        skill_tools = self._skill_tools
        skill_funcs = self._skill_funcs
        guidance_parts = self._prompt_guidance_parts
        combined_guidance = None
        if guidance_parts:
            combined_guidance = '\n'.join((f'- {g}' for g in guidance_parts))
        self._original_agent_init = LLMAgent.__init__
        self._original_system_prompt = LLMAgent.system_prompt
        _orig_init = self._original_agent_init
        _orig_prompt = self._original_system_prompt

        def patched_init(self_agent, tools: list, domain_policy: str, **kwargs):
            tools = list(tools) + list(skill_tools.values())
            _orig_init(self_agent, tools=tools, domain_policy=domain_policy, **kwargs)

        def patched_system_prompt(self_agent):
            base = _orig_prompt.fget(self_agent)
            if combined_guidance:
                base += '\n<skills>\nYou have access to the following additional skill tools:\n' + combined_guidance + '\n</skills>'
            return base
        LLMAgent.__init__ = patched_init
        LLMAgent.system_prompt = property(patched_system_prompt)
        self._original_use_tool = ToolKitBase.use_tool
        self._original_has_tool = ToolKitBase.has_tool
        _orig_use = self._original_use_tool
        _orig_has = self._original_has_tool

        def patched_use_tool(self_tk, tool_name: str, **kwargs):
            if tool_name in skill_funcs:
                return skill_funcs[tool_name](**kwargs)
            return _orig_use(self_tk, tool_name, **kwargs)

        def patched_has_tool(self_tk, tool_name: str) -> bool:
            if tool_name in skill_funcs:
                return True
            return _orig_has(self_tk, tool_name)
        ToolKitBase.use_tool = patched_use_tool
        ToolKitBase.has_tool = patched_has_tool
        self._applied = True

    def restore(self):
        if not self._applied:
            return
        LLMAgent.__init__ = self._original_agent_init
        LLMAgent.system_prompt = self._original_system_prompt
        ToolKitBase.use_tool = self._original_use_tool
        ToolKitBase.has_tool = self._original_has_tool
        self._applied = False

    def get_skill_names(self) -> List[str]:
        return list(self._skill_funcs.keys())

    def get_summary(self) -> str:
        lines = [f'SkillInjector: {len(self._skill_funcs)} skills loaded']
        for (name, meta) in self._skill_metadata.items():
            lines.append(f"  - {name} ({meta.get('domain', '?')}): {meta.get('description', '')[:60]}")
        return '\n'.join(lines)

    @staticmethod
    def extract_tool_traces(simulation) -> list:
        traces = []
        for (i, msg) in enumerate(simulation.messages):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    traces.append({'step': i, 'tool': tc.name if hasattr(tc, 'name') else tc.get('name', '?'), 'args': tc.arguments if hasattr(tc, 'arguments') else tc.get('arguments', {})})
        return traces
