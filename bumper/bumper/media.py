""" media.py
Class for  media queries (e.g., pdfs, images, etc.)
"""

from openai import OpenAI
from . import threadb
from . import paths
from .openai import MODELS

__all__ = ["MediaAssistant"]

media_specs = []
media_functions = {}

class MediaAssistant:
    def __init__(self, name = "BUMPER media assistant", model=MODELS.gpt4, max_prompt_tokens=5000, vector_store_id_file:str=''):
        # settings
        self.client = OpenAI()
        self.name = name
        self.instructions = "You are a helpful assistant."
        self.model = model
        self.tools = [{"type": "file_search"}]
        self.max_prompt_tokens = max_prompt_tokens
        self.vector_store_id_file = vector_store_id_file
        self._create()

        # setup thread database
        self.threadb = threadb.Threadb(db_name="mediaAssistant.db")
        self.threads = []

    def _create(self):
        """ Create the openai assistant """
        self.assistant = self.client.beta.assistants.create(
            name=self.name,
            instructions=self.instructions,
            tools=self.tools,
            model=self.model,
        )

        # get the vectore store id values from paths.data / "vector-store-evidence.id"
        with open(paths.data / self.vector_store_id_file, "r") as f:
            vector_store_ids = f.readlines()            

        # add to the assistant
        self.assistant = self.client.beta.assistants.update(
            assistant_id=self.assistant.id,
            tool_resources={"file_search": {"vector_store_ids": vector_store_ids}},
        )
        
    def create_thread(self):
        """ Create a thread """
        # create thread (set as current)
        self.thread = self.client.beta.threads.create()
        # save thread
        self.threads.append(self.thread)
        # save thread to the database
        self.threadb.add_thread(self.thread.id)

    def launch(self, message, new_thread=True):
        """ Launch the query without streaming """
        if new_thread:
            self.create_thread()
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id, role="user", content=message
        )

        self.run = self.client.beta.threads.runs.create_and_poll(
        thread_id=self.thread.id,
        assistant_id=self.assistant.id,
        max_prompt_tokens=self.max_prompt_tokens
        )

    def get_messages(self):
        """ Get the messages """
        return self.client.beta.threads.messages.list(thread_id=self.thread.id)
    
    def cleanup(self):
        """ Cleanup the assistant """
        self.client.beta.assistants.delete(self.assistant.id)
        self.threadb.delete_all_threads()

#########################################

def simple_query(message, assistant):
    """ Query the evidence/model to learn about the methodology and applications """
    assistant.launch(message)
    msg =  assistant.get_messages()
    return msg.data[0].content[0].text.value

media_functions["simple_query"] = simple_query
media_specs.append(
    {
        "type": "function",
        "function": {
            "name": "simple_query",
            "description": simple_query.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Copy the query message exactly.",
                    },
                    "unit": {"type": "string"},
                },
                "required": ["message"],
            },
        },
    }
)