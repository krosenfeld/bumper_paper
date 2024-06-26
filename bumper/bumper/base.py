""" base.py
Base classes
"""


import numpy as np
from openai import OpenAI
from typing_extensions import override

from . import threadb
from .openai import MODELS
# from .media import media_specs, media_functions, MediaAssistant
# from .tools import tool_specs, tool_functions
# from .rugby_tools import RUGBY_TOOLS
from .utils import pretty_print_conversation
from .judges import ElementJudge

__all__ = ["bump", "BaseBumper"]


def bump(assistant, question, new_thread=True, n=1):
    """ BUMPER workflow """
    if new_thread:
        assistant.create_thread()
    assistant.message(question)
    assistant.launch_message()
    if assistant.get_status() != "completed":
        print(f"status: {assistant.get_status()}")
        assistant.launch_tools()
    else:
        # no tools used, raise error
        raise RuntimeError("No tools found, query is out of scope")
    if assistant.get_status() == "completed":
        # count tokens
        assistant.update_thread_tokens()
        probs, nan_count = assistant.launch_judgement(n=n)
        assistant.print_messages()
        return probs, nan_count
    else :
        raise RuntimeError("Assistant not completed")


class BaseBumper:
    def __init__(self, 
        model=MODELS.gpt4,
        checks=True,        
        sop_file=None,
        max_prompt_tokens=20000,
        explain=True,
        verbose=False,
        judge=None,
        name = "BaseBumper",
        tools = []):

        self.client = OpenAI()
        self.model = model
        self.name = name
        self.instructions = "You are a helpful assistant. Use the provided tools and functions to answer questions. Try not to extrapolate."
        self.tools = tools
        self.checks = checks
        self.max_prompt_tokens = max_prompt_tokens
        self.sop_file = sop_file
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.verbose = verbose
        self.explain = explain

        self.sub_assistants = {}
        self.available_functions = {}

        # add judge
        if judge is None:
            self.judge = ElementJudge(
                model=self.model, sop_file=self.sop_file, explain=self.explain
            )
        else:
            self.judge = judge

        # create a thread
        self.thread = None
        self.threadb = threadb.Threadb(db_name=f"{self.name}.db")
        self.threads = []

    def _create(self):
        """Create the openai assistant"""
        self.assistant = self.client.beta.assistants.create(
            name=self.name,
            instructions=self.instructions,
            tools=self.tools,
            model=self.model,
        )


    def create_thread(self):
        """Create a thread"""
        # create thread (set as current)
        self.thread = self.client.beta.threads.create()
        # save thread
        self.threads.append(self.thread)
        # save thread to the database
        self.threadb.add_thread(self.thread.id)

    def message(self, msg, role="user", thread=None):
        """Send message to the assistant"""
        if thread is None:
            thread = self.thread
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id, role=role, content=msg
        )
        return message

    def launch_message(self):
        """Launch the query without streaming"""
        self.run = self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            max_prompt_tokens=self.max_prompt_tokens,
        )

    def get_messages(self):
        """Get the messages"""
        return self.client.beta.threads.messages.list(thread_id=self.thread.id)

    def get_status(self):
        """Get the status of the run"""
        return self.run.status

    def print_messages(self):
        """Print the messages"""
        pretty_print_conversation(self.get_messages())

    def launch_tools(self):
        pass

    def update_thread_tokens(self, thread_id=None):
        if thread_id is None:
            thread_id = self.thread.id
        runs = self.client.beta.threads.runs.list(thread_id=thread_id)
        self.completion_tokens += runs.data[0].usage.completion_tokens
        self.prompt_tokens += runs.data[0].usage.prompt_tokens

    def launch_judgement(self, n=3, model=None):
        """Launch checks to the most recent message"""
        msg = self.get_messages().data[0].content[0].text.value
        probs = []
        nan_count = 0 
        for i in range(n):
            call_probs, call_nan_count = self.judge.rules(msg)
            probs += call_probs
            nan_count += call_nan_count
        probs = np.array(probs)

        MSG = f"{n} trials: {np.sum(probs > 0)} passes, {np.sum(np.logical_and(probs < 0, probs >= -100))} fails, {nan_count} parse errors (P={probs})"

        self.message(MSG, role="assistant")
        return probs, nan_count

    def summarize_tokens(self):
        prompt_tokens = self.prompt_tokens + self.judge.prompt_tokens
        completion_tokens = self.completion_tokens + self.judge.completion_tokens
        print(
            f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {prompt_tokens + completion_tokens}"
        )

    def cleanup(self):
        """Cleanup the assistant"""

        for _, sub_assistant in self.sub_assistants.items():
            sub_assistant.cleanup()

        self.client.beta.assistants.delete(self.assistant.id)
        self.threadb.delete_all_threads()
