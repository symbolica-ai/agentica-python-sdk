"""
Running agentica as a script drops you into a TUI REPL with the Agentica SDK loaded.
"""

import argparse
import asyncio
import json
import os
import signal
import sqlite3
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Any, AsyncIterator, Literal

from agentica_internal.core.ansi import palettes
from agentica_internal.warpc import forbidden
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style as PTKStyle
from pygments import token as pt
from rich import syntax
from rich._loop import loop_last
from rich.color import Color, blend_rgb
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.live_render import LiveRender
from rich.markdown import CodeBlock, Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from rich.theme import Theme

from agentica import ModelStrings
from agentica.common import Chunk, UserRole
from agentica.demo.chat_with import ChatWith
from agentica.errors import InsufficientCreditsError, InvalidAPIKey, ServerError
from agentica.logging import set_default_agent_listener


class TailLiveRender(LiveRender):
    """LiveRender that shows the tail (bottom) of content when it overflows, not the top."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        renderable = self.renderable
        style = console.get_style(self.style)
        lines = console.render_lines(renderable, options, style=style, pad=False)
        shape = Segment.get_shape(lines)

        _, height = shape
        if height > options.size.height:
            # Take the LAST N lines instead of first N (tail mode)
            lines = lines[-options.size.height :]
            shape = Segment.get_shape(lines)
        self._shape = shape

        new_line = Segment.line()
        for last, line in loop_last(lines):
            yield from line
            if not last:
                yield new_line


class TailLive(Live):
    """Live display that shows the tail (bottom) of content when it overflows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the LiveRender with our TailLiveRender
        self._live_render = TailLiveRender(
            self.get_renderable(), vertical_overflow=self.vertical_overflow
        )

    def stop(self) -> None:
        """Stop live rendering - override to NOT switch to visible on final render.

        Rich's default stop() sets vertical_overflow="visible" before the final refresh,
        which causes scrollback pollution. We keep cropping to avoid this.
        """
        with self._lock:
            if not self._started:
                return
            self._started = False
            self.console.clear_live()

            if self.auto_refresh and self._refresh_thread is not None:
                self._refresh_thread.stop()
                self._refresh_thread = None

            # NOTE: We intentionally do NOT set self.vertical_overflow = "visible" here
            # to prevent scrollback pollution. The caller will print full content after.

            with self.console:
                try:
                    if not self._alt_screen and not self.console.is_jupyter:
                        self.refresh()
                finally:
                    self._disable_redirect_io()
                    self.console.pop_render_hook()
                    # Only add newline if NOT transient - otherwise it leaves one line behind
                    if not self._alt_screen and self.console.is_terminal and not self.transient:
                        self.console.line()
                    self.console.show_cursor(True)
                    if self._alt_screen:
                        self.console.set_alt_screen(False)
                    if self.transient and not self._alt_screen:
                        self.console.control(self._live_render.restore_cursor())
                    if self.ipy_widget is not None and self.transient:
                        self.ipy_widget.close()


from dotenv import load_dotenv

load_dotenv()

set_default_agent_listener(None)


BILLING_URL = "https://platform.symbolica.ai/billing"
BILLING_LINK = f"[link={BILLING_URL}]{BILLING_URL}[/link]"

DEEP_LINK_URL = "https://platform.symbolica.ai/api-keys"
DEEP_LINK = f"[link={DEEP_LINK_URL}]{DEEP_LINK_URL}[/link]"

DEFAULT_AGENT_MODEL: ModelStrings = 'anthropic:claude-sonnet-4.5'


# Config file for persisting preferences
CONFIG_DIR = Path.home() / '.config' / 'agentica'
CONFIG_FILE = CONFIG_DIR / 'config.json'


def _load_config() -> dict:
    """Load config from file. Returns empty dict on any error."""
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_config(config: dict) -> None:
    """Save config to file. Silent on any error."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(config, indent=2))
    except Exception:
        pass


def get_saved_theme() -> Literal['light', 'dark'] | None:
    """Get saved theme preference, or None if not set."""
    config = _load_config()
    theme = config.get('theme')
    if theme in ('light', 'dark'):
        return theme  # type: ignore
    return None


def save_theme(theme: Literal['light', 'dark']) -> None:
    """Save theme preference to config."""
    config = _load_config()
    config['theme'] = theme
    _save_config(config)


PERIWINKLE = palettes.Agentica.light.hex_str
PERIWINKLE_DARK = palettes.Agentica.dark.hex_str

term_theme: Literal['light', 'dark'] = 'dark'
console = c = Console()


def syntax_style(theme: Literal['light', 'dark']) -> dict[Any, str]:
    """Generate pygments style dict with appropriate text color for theme."""
    text_color = '#000000' if theme == 'light' else '#FFFFFF'
    return {
        pt.Comment: f"italic {text_color}",
        pt.Comment.Preproc: f"noitalic {text_color}",
        pt.Keyword: f"bold {text_color}",
        pt.Keyword.Pseudo: f"nobold {text_color}",
        pt.Keyword.Type: f"nobold {text_color}",
        pt.Operator.Word: f"bold {text_color}",
        pt.Name.Class: f"bold {text_color}",
        pt.Name.Namespace: f"bold {text_color}",
        pt.Name.Exception: f"bold {text_color}",
        pt.Name.Entity: f"bold {text_color}",
        pt.Name.Tag: f"bold {text_color}",
        pt.String: f"italic {text_color}",
        pt.String.Interpol: f"bold {text_color}",
        pt.String.Escape: f"bold {text_color}",
        pt.Generic.Heading: f"bold {text_color}",
        pt.Generic.Subheading: f"bold {text_color}",
        pt.Generic.Emph: f"italic {text_color}",
        pt.Generic.Strong: f"bold {text_color}",
        pt.Generic.EmphStrong: f"bold italic {text_color}",
        pt.Generic.Prompt: f"bold {text_color}",
        pt.Error: f"border:#FF0000 {text_color}",
        # Catch-all for any token type
        pt.Token: text_color,
    }


def blend(color1_: str, color2_: str, f: float = 0.5) -> str:
    color1 = Color.parse(color1_)
    color2 = Color.parse(color2_)
    b = blend_rgb(color1.get_truecolor(), color2.get_truecolor(), f)
    return f'#{b[0]:02X}{b[1]:02X}{b[2]:02X}'


def fade(c: str, f: float = 0.5) -> str:
    return blend(c, '#FFFFFF' if term_theme == 'light' else '#000000', f)


def set_theme(theme: Literal['light', 'dark']):
    global \
        term_theme, \
        console, \
        c, \
        HIGHLIGHT, \
        FADED, \
        CONTRASTING, \
        HL_STYLE, \
        HL_DIM_STYLE, \
        HLB_STYLE, \
        HLI_STYLE, \
        HLBG_STYLE, \
        HLBG_BOLD_STYLE, \
        PeriwinkleStyle, \
        PlainStyle

    term_theme = theme
    is_light = theme == 'light'

    HIGHLIGHT = PERIWINKLE if is_light else PERIWINKLE_DARK
    CONTRASTING = PERIWINKLE_DARK if is_light else PERIWINKLE
    FADED = '#CCCCCC' if is_light else '#999999'

    HL_STYLE = Style(color=CONTRASTING)
    HL_DIM_STYLE = Style(color=CONTRASTING, dim=True)
    HLB_STYLE = Style(color=CONTRASTING, bold=True)
    HLI_STYLE = Style(color=CONTRASTING, italic=True)
    HLBG_STYLE = Style(color='#FFFFFF', bgcolor=CONTRASTING)
    HLBG_BOLD_STYLE = Style(color='#FFFFFF', bgcolor=CONTRASTING, bold=True)

    base_color = '#FFFFFF' if is_light else '#333333'
    peri_bg = blend(HIGHLIGHT, base_color, 0.5 if is_light else 0.25)
    plain_bg = blend(HIGHLIGHT, base_color, 0.92 if is_light else 0.75)

    class PeriwinkleStyle(syntax.PygmentsStyle):
        name = 'periwinkle'
        background_color = peri_bg
        styles = syntax_style(theme)

    class PlainStyle(syntax.PygmentsStyle):
        name = 'plain'
        background_color = plain_bg
        styles = syntax_style(theme)

    code_text_color = '#000000' if is_light else '#FFFFFF'
    code_bg = blend(HIGHLIGHT, '#FFFFFF' if is_light else '#000000', 0.85 if is_light else 0.25)

    console = c = Console(
        theme=Theme(
            styles={
                # Our custom styles
                'hl': HL_STYLE,
                'hlb': HLB_STYLE,
                'hli': HLI_STYLE,
                'hlbg': HLBG_STYLE,
                'hlbgb': HLBG_BOLD_STYLE,
                'rule.line': HLB_STYLE,
                'status.spinner': HLB_STYLE,
                # Markdown overrides
                'markdown.block_quote': HL_STYLE,
                'markdown.code': Style(color=code_text_color, bold=False, bgcolor=code_bg),
                'markdown.code_block': Style(color=code_text_color, bgcolor=code_bg),
                'markdown.link': HL_STYLE,
                'markdown.link_url': Style(color=CONTRASTING, underline=True),
                'markdown.item.bullet': HLB_STYLE,
                'markdown.item.number': HLB_STYLE,
                'markdown.hr': HL_STYLE,
                # Override repr/highlight styles to use periwinkle instead of magenta/yellow
                'repr.call': HLB_STYLE,
                'repr.attrib_name': HL_STYLE,
                'repr.attrib_value': HL_STYLE,
                'repr.none': Style(color=CONTRASTING, italic=True),
                'repr.path': HL_STYLE,
                'repr.filename': HLB_STYLE,
                'repr.number': Style(color=CONTRASTING, bold=True),
                'repr.bool_true': Style(color=CONTRASTING, italic=True),
                'repr.bool_false': Style(color=CONTRASTING, italic=True, dim=True),
                'repr.str': Style(color=CONTRASTING),
                'repr.tag_name': HLB_STYLE,
                'repr.url': Style(color=CONTRASTING, underline=True),
                'inspect.attr': Style(color=CONTRASTING, italic=True),
                'inspect.callable': HLB_STYLE,
                'json.key': HLB_STYLE,
                'json.str': HL_STYLE,
                'json.number': Style(color=CONTRASTING, bold=True),
                'json.bool_true': Style(color=CONTRASTING, italic=True),
                'json.bool_false': Style(color=CONTRASTING, italic=True, dim=True),
                'json.null': Style(color=CONTRASTING, italic=True),
                'prompt.choices': HLB_STYLE,
                'prompt.default': HL_STYLE,
            },
            inherit=True,
        )
    )


# default to dark theme
set_theme('dark')

INFO = '[hlb]*[/hlb]'
WARN = '[hlb]![/hlb]'
PROMPT_PREFIX: FormattedText = [('class:prompt', '> ')]

# Available models
AVAILABLE_MODELS = [
    'openai:gpt-3.5-turbo',
    'openai:gpt-4o',
    'openai:gpt-4.1',
    'openai:gpt-5',
    'anthropic:claude-sonnet-4',
    'anthropic:claude-opus-4.1',
    'anthropic:claude-sonnet-4.5',
    'anthropic:claude-opus-4.5',
]

# Reuse prompt session for history across prompts
_prompt_session: PromptSession[str] | None = None
HISTORY_FILE = CONFIG_DIR / 'history'


def _get_prompt_session() -> PromptSession[str]:
    global _prompt_session
    if _prompt_session is None:
        # Ensure config directory exists for history file
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _prompt_session = PromptSession(history=FileHistory(str(HISTORY_FILE)))
    return _prompt_session


class StartOfStringCompleter(Completer):
    """Complete based on the full input from the start, allowing spaces."""

    def __init__(self, words: list[str]):
        self.words = words

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        for word in self.words:
            if word.startswith(text) and word != text:
                yield Completion(word, start_position=-len(text))


def _clear_lines_above(n: int) -> None:
    """Clear n lines above the cursor and leave cursor at the top cleared line."""
    # Rich doesn't provide line-clearing, so we use standard ANSI sequences
    # ESC[F = move up one line, ESC[2K = clear entire line
    for _ in range(n):
        c.file.write('\x1b[F\x1b[2K')
    c.file.flush()


def _ptk_style() -> PTKStyle:
    """Get prompt_toolkit style matching current theme."""
    return PTKStyle.from_dict(
        {
            'prompt': f'bold {HIGHLIGHT}',
            'placeholder': f'italic {FADED}',
        }
    )


async def simple_input(placeholder: str = "", clear: bool = False) -> str:
    """Simple input with just a placeholder, no completions."""
    session = _get_prompt_session()
    ph: FormattedText | None = [('class:placeholder', placeholder)] if placeholder else None

    result = await session.prompt_async(PROMPT_PREFIX, placeholder=ph, style=_ptk_style())

    if clear:
        _clear_lines_above(1)

    return result


async def wait_for_enter() -> None:
    """Display '... Enter to continue' and wait for enter key only."""
    # Use a separate session to avoid interfering with the main prompt session
    session = PromptSession()

    # Key bindings: only Enter does anything (accepts), all else ignored
    kb = KeyBindings()

    @kb.add(Keys.Enter)
    def accept(event):
        event.app.exit(result='')

    @kb.add(Keys.Any)
    def ignore(event):
        pass  # Ignore all other keys

    # Dim italic style for the prompt
    style = PTKStyle.from_dict(
        {
            'prompt': f'italic {FADED}',
        }
    )

    prompt_text: FormattedText = [('class:prompt', '... Enter to continue')]

    await session.prompt_async(prompt_text, style=style, key_bindings=kb)

    # Clear the prompt line
    _clear_lines_above(1)


class AgenticaCodeBlock(CodeBlock):
    _original_lexer_name: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_lexer_name = self.lexer_name

    @classmethod
    def create(cls, markdown: Markdown, token) -> CodeBlock:
        node_info = token.info or ""
        lexer_name = node_info.partition(" ")[0]

        code_theme = PlainStyle
        if lexer_name == 'python':
            code_theme = PeriwinkleStyle

        self = cls(lexer_name or 'python', code_theme)  # type: ignore
        self._original_lexer_name = lexer_name
        return self

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self._original_lexer_name == 'python':
            yield Text('EXECUTE', style=HL_DIM_STYLE)
        if self._original_lexer_name == 'output':
            yield Text('OUTPUT', style=HL_DIM_STYLE)
        code = str(self.text).rstrip()
        s = syntax.Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            word_wrap=True,
            padding=1,
        )
        yield s


class AgenticaMarkdown(Markdown):
    # - allow replacing interior parsed text
    # - highlight all text with periwinkle when appropriate
    # - ```python code blocks get a periwinkle background and start with the word EXECUTE
    # - plain code blocks get a dim background and python syntax highlighting
    # - blockquotes get periwinkle accents

    elements = {
        **Markdown.elements,
        'fence': AgenticaCodeBlock,
        'code_block': AgenticaCodeBlock,
    }

    def replace(self, markup: str) -> None:
        self.__init__(markup)


@contextmanager
def status(message: str = ""):
    with c.status(message, spinner='star'):
        yield


def append_to_dotenv(key: str, value: str):
    prefix = "" if not os.path.exists('.env') else "\n"
    with open('.env', 'a') as f:
        f.write(f"{prefix}{key}={value}\n")


async def api_key_flow(force: bool = False):
    """If the user needs an API key, send them a deep link to generate it and ask fro the key."""
    if not force and os.getenv('AGENTICA_API_KEY'):
        return

    if not force:
        c.print()
        c.print(
            WARN,
            f"No [bold]API key[/bold] found, please generate one at: {DEEP_LINK}",
        )
        c.print()
        c.print("Once that's done, input your API key here:")
    else:
        c.print()
        c.print(f"You may generate a new API key here: {DEEP_LINK}")
        c.print()
        c.print("Please input your API key here:")

    while True:
        api_key = await simple_input(placeholder="symbolica_xxx...xxx", clear=True)
        os.environ['AGENTICA_API_KEY'] = api_key

        # Try to authenticate the API key
        from agentica.agentica_client.global_csm import get_global_csm

        try:
            _ = await get_global_csm()
            break
        except InvalidAPIKey:
            c.print(WARN, "Invalid API key.")
            os.environ.pop('AGENTICA_API_KEY', None)

    with status("Saving to .env..."):
        append_to_dotenv('AGENTICA_API_KEY', api_key)
        await asyncio.sleep(0.5)
    c.print()
    c.print(INFO, "Saved your [bold]AGENTICA_API_KEY[/bold] to [italic].env[/italic]")
    c.print()


def force_local():
    os.environ['S_M_BASE_URL'] = 'http://localhost:2345'
    os.environ.pop('AGENTICA_API_KEY', None)
    os.environ.pop('AGENTICA_BASE_URL', None)


class StreamInterrupted(Exception):
    """Raised when streaming is interrupted by Ctrl+C."""

    pass


async def stream_chunks(stream: AsyncIterator[Chunk], no_md: bool = False) -> str:
    """Stream the outputted chunks from an invocation."""
    md_accum = ""
    last_role = None
    interrupted = False

    # Set up signal handler to set interrupted flag
    loop = asyncio.get_event_loop()

    def on_sigint():
        nonlocal interrupted
        interrupted = True

    loop.add_signal_handler(signal.SIGINT, on_sigint)

    def md_add(txt: str):
        nonlocal md_accum
        md_accum += txt

    def print_add(txt: str):
        nonlocal md_accum
        md_accum += txt
        print(txt, end='', flush=True)

    add = print_add if no_md else md_add

    def process_chunk(chunk: Chunk):
        nonlocal last_role
        role = chunk.role
        changed_role = role != last_role and last_role is not None
        last_role = role

        if changed_role:
            add("\n\n")

        if role == 'agent':
            add(chunk.content)
        else:
            is_exec = (
                'execution' in role.role
                or (isinstance(role, UserRole) and 'execution' in (role.name or ''))
                or '<execution>' in chunk.content
            )
            if not is_exec:
                return
            content = chunk.content
            content = content.replace('<execution>', '')
            content = content.replace('</execution>', '')
            if (lines := content.split('\n')) and len(lines) > 10:
                content = '\n'.join(lines[:10]) + '\n [...]'
            content = content.strip()

            add('\n````output\n' + content + '\n````\n')

    try:
        if no_md:
            async for chunk in stream:
                if interrupted:
                    break
                process_chunk(chunk)
        else:
            md = AgenticaMarkdown(md_accum)
            with TailLive(
                refresh_per_second=12.5,
                console=console,
                transient=True,
                vertical_overflow="crop",
            ) as live:
                async for chunk in stream:
                    if interrupted:
                        break
                    process_chunk(chunk)
                    try:
                        md.replace(md_accum)
                        live.update(md)
                    except:
                        # Catch any markdown parsing errors while we stream.
                        live.console.print(f"[red](failed to render)[/red]")
            # Print blank line to cover any edge-case leftover, then full markdown
            if not interrupted:
                c.print()
                c.print(md)

    finally:
        # Restore default signal handler
        loop.remove_signal_handler(signal.SIGINT)

    if interrupted:
        c.print()
        c.print("[dim]Interrupted[/dim]")
        raise StreamInterrupted()

    return md_accum


async def prompt(
    label: str = "",
    placeholder: str = "",
    suggestions: list[str] | None = None,
    first_word_only: bool = False,
) -> str:
    """
    Display a styled input prompt with:
    - Rule above
    - "> " prefix in periwinkle
    - Placeholder text (grayed out)
    - Word completions for commands (Tab to complete)

    The entire input area disappears after submission.
    """
    session = _get_prompt_session()

    if suggestions:
        completer: Completer | None = (
            StartOfStringCompleter(suggestions)
            if first_word_only
            else WordCompleter(suggestions, ignore_case=True)
        )
    else:
        completer = None

    # Top rule via Rich
    c.rule(label, style="hl")

    ph: FormattedText | None = [('class:placeholder', placeholder)] if placeholder else None
    result = await session.prompt_async(
        PROMPT_PREFIX, placeholder=ph, completer=completer, style=_ptk_style()
    )

    # Clear input area (top rule + input line)
    _clear_lines_above(2)

    return result


async def select_model(current_model: str) -> str:
    """
    Display a model picker with suggestions.
    Returns the selected model name.
    """
    session = _get_prompt_session()

    # Print model list (we'll clear this after selection)
    c.print("[bold]Available models:[/bold]")
    for model in AVAILABLE_MODELS:
        prefix = "[bold] * [/bold]" if model == current_model else "   "
        c.print(f"{prefix}[normal]{model}[/normal]", highlight=False)
    c.print("[dim italic]   ... or input any OpenRouter model slug[/dim italic]")
    model_list_lines = len(AVAILABLE_MODELS) + 2  # +2 for header and openrouter note

    c.rule("Select Model", style="hl")

    ph: FormattedText = [('class:placeholder', f"Current: {current_model}")]
    completer = WordCompleter(AVAILABLE_MODELS, ignore_case=True)
    result = await session.prompt_async(
        PROMPT_PREFIX, placeholder=ph, completer=completer, style=_ptk_style()
    )

    # Clear the model list + rule + input line
    _clear_lines_above(model_list_lines + 2)

    return result.strip() if result.strip() else current_model


def _is_bad_model_error(e: ServerError, model: str) -> bool:
    """Check if the error is about an invalid model."""
    msg = str(e).lower()
    return any(x in msg for x in ['model', 'badmodel', model.lower(), 'not available'])


class Fail(Enum):
    BadAPIKey = "bad API key"
    BadModel = "bad model name"


async def create_agent(model: str) -> ChatWith | Fail:
    """Create a ChatWith agent with a loading spinner. Returns None if model is invalid."""

    # Extra tools:
    database = sqlite3.connect(':memory:')

    def execute_sql(sql: str) -> list[Any]:
        """execute + fetchall on the database"""
        return database.execute(sql).fetchall()

    extra_tools = {
        'execute_sql': execute_sql,
    }

    forbidden.whitelist_objects(execute_sql)

    extra_premise = f"""
    You are backed by the {model} model.

    You also have access to a sample tool: `execute_sql`, which allows you to execute SQL queries on a in-memory database for demonstration purposes.
    The database starts off as **empty**.

    ```
    extra_tools = {{ 'execute_sql': execute_sql }}
    ```

    Up front, here are some questions the human might ask you:
    """
    extra_premise = dedent(extra_premise).strip()
    extra_premise += '\n' + '\n'.join(f" - {p}" for p in EXAMPLE_PROMPTS)

    try:
        with status(f"Loading agent with [hlb]{model}[/hlb]..."):
            return await ChatWith.create(
                model=model, extra_tools=extra_tools, extra_premise=extra_premise
            )
    except InvalidAPIKey:
        c.print(WARN, "Invalid API key")
        return Fail.BadAPIKey
    except ServerError as e:
        if _is_bad_model_error(e, model):
            c.print(
                f"[bold]Invalid model:[/bold] [italic]{model}[/italic] is not available.",
                highlight=False,
            )
            return Fail.BadModel
        raise


async def ask_theme() -> Literal['light', 'dark']:
    session = _get_prompt_session()
    ph: FormattedText = [('class:placeholder', 'light / dark')]
    completer = WordCompleter(['light', 'dark'], ignore_case=True)

    c.print("Is your terminal light or dark theme?")

    while True:
        result = await session.prompt_async(
            PROMPT_PREFIX, placeholder=ph, completer=completer, style=_ptk_style()
        )
        result = result.strip().lower()

        if result in ('light', 'dark'):
            # Clear the prompt line + question line
            _clear_lines_above(2)
            c.print(f"Theme set to [hl]{result}[/hl]")
            save_theme(result)  # type: ignore
            return result  # type: ignore

        # Invalid input - clear and try again
        _clear_lines_above(1)
        c.print(f"[red]Invalid:[/red] please enter 'light' or 'dark'")


EXAMPLE_PROMPTS = [
    "What makes Agentica different from other agent frameworks?",
    "Can you show me the definition of spawn()?",
    "Explain spawn() to me with an example.",
    "Can you show me the definition of execute_sql()?",
    "Populate the database with random shop-like data.",
    "Show me what the database currently looks like.",
    "How are you able to use my own code and data structures?",
    "How are you capable of returning arbitrary data types?",
]


async def fake_intro():
    c.print("> Introduce yourself", style='dim')

    with status():
        await asyncio.sleep(0.2)

    c.print()

    intro = """
    Hey! I'm an agent created with the Agentica framework.

    Would you like me to see me demonstrate any of the following?

    [hlb]A[/hlb]: Tool use without MCP
    [hlb]B[/hlb]: Multi agent orchestration
    [hlb]C[/hlb]: How I, as an Agentica agent, have been defined and created?

    [italic]or[/italic] I can answer any questions you may have about the Agentica framework.
    """

    intro = dedent(intro).strip()
    chunks = intro.split(' ')

    for chunk in chunks:
        c.print(chunk, highlight=False, end=' ')
        await asyncio.sleep(0.04)

    c.print()


async def main():
    args = parse_args()
    current_model = args.model

    know_theme = False
    if args.light or args.dark:
        theme = 'light' if args.light else 'dark'
        set_theme(theme)
        c.print(f"Theme set to [hl]{theme}[/hl]")
        know_theme = True
    elif saved_theme := get_saved_theme():
        set_theme(saved_theme)
        know_theme = True

    c.print(
        Panel.fit("[hlb]▣[/hlb]  Welcome to the [hlb]Interactive Agentica Docs[/hlb]", style="bold")
    )

    if not know_theme:
        theme = await ask_theme()
        set_theme(theme)
        save_theme(theme)

    if not args.local:
        await api_key_flow()
    else:
        force_local()

    c.print()

    # Create the initial agent
    agent = await create_agent(current_model)
    while isinstance(agent, Fail):
        if agent == Fail.BadAPIKey:
            await api_key_flow(force=True)
            agent = await create_agent(current_model)
        elif agent == Fail.BadModel:
            c.print("Please select a valid model.")
            current_model = await select_model(current_model)
            agent = await create_agent(current_model)
        else:
            raise ValueError(f"Unknown failure: {agent}")

    c.print(INFO, f"Using model: [hlb]{current_model}[/hlb]")
    c.print()

    if not args.no_intro:
        c.rule(style="dim")
        c.print()
        await fake_intro()
        c.print()

    # Main chat loop
    while True:
        try:
            user_input = await prompt(
                label="Chat",
                placeholder="Type your message...",
                suggestions=["/quit", "/model", "/theme", "/help"] + EXAMPLE_PROMPTS,
                first_word_only=True,
            )
        except (KeyboardInterrupt, EOFError):
            c.print()
            c.print("[dim]Goodbye![/dim]")
            break

        # Handle commands
        if user_input.strip() == "/quit":
            c.print("[dim]Goodbye![/dim]")
            break

        if user_input.strip() == "/model":
            new_model = await select_model(current_model)
            if new_model != current_model:
                new_agent = await create_agent(new_model)
                if new_agent is not None:
                    current_model = new_model
                    agent = new_agent
                    c.print(INFO, f"Switched to model: [hlb]{current_model}[/hlb]")
                    c.print()
                # If None, error was already printed, keep current model
            continue

        if user_input.strip() == "/theme":
            new_theme: Literal['light', 'dark'] = 'light' if term_theme == 'dark' else 'dark'
            set_theme(new_theme)
            save_theme(new_theme)
            c.print(INFO, f"Switched to [hlb]{new_theme}[/hlb] theme")
            c.print()
            continue

        if user_input.strip() == "/help":
            c.print("[hlb]/quit[/hlb]  - Exit the chat")
            c.print("[hlb]/model[/hlb] - Change the model")
            c.print("[hlb]/theme[/hlb] - Toggle light/dark theme")
            c.print()
            continue

        assert isinstance(agent, ChatWith)

        if not user_input.strip():
            continue

        # Show what user typed
        c.print(f"[dim]>[/dim] {user_input}")
        c.print()

        # Chat with the agent
        try:
            res, first_chunk, stream = agent.chat(user_input)
            with status():
                _ = await first_chunk
            _ = await stream_chunks(stream, no_md=args.no_md)
            if response := await res:
                c.print(Markdown(md_quote(response)))

            c.print()
        except InsufficientCreditsError:
            c.print(WARN, "Insufficient credits. Please add more credits to your account!")
            c.print(WARN, f"You can top-up at: {BILLING_LINK}")
        except StreamInterrupted:
            # Already printed "Interrupted" in stream_chunks
            c.print()


def md_quote(text: str) -> str:
    """Quote the text with markdown."""
    return '\n> ' + '\n> '.join(text.split('\n'))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Agentica SDK REPL')
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_AGENT_MODEL,
        help='The model to use for the agent',
    )
    parser.add_argument(
        '--local',
        action='store_true',
        default=False,
        help='Whether to run locally (i.e. no API key required)',
    )
    parser.add_argument(
        '--no-md',
        action='store_true',
        default=False,
        help='Whether to disable markdown rendering',
    )
    parser.add_argument(
        '--light',
        action='store_true',
        default=False,
        help='Indicate your terminal is using a light theme',
    )
    parser.add_argument(
        '--dark',
        action='store_true',
        default=False,
        help='Indicate your terminal is using a dark theme',
    )
    parser.add_argument(
        '--no-intro',
        action='store_true',
        default=False,
        help='Skip the introductory agent prompt',
    )
    return parser.parse_args()


def cli():
    """Sync entry point for console script."""

    async def safe_main():
        try:
            await main()
        except (KeyboardInterrupt, EOFError):
            c.print()
            c.print("[dim]Goodbye![/dim]")

    asyncio.run(safe_main())


forbidden.whitelist_modules('agentica.__main__')


if __name__ == "__main__":
    cli()
