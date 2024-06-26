from openai import OpenAI

# class to access the OpenAI models
class OpenAIModels:
    gpt3 = "gpt-3.5-turbo-0125"
    gpt4 = "gpt-4-0125-preview"
    gpt4o = "gpt-4o-2024-05-13"
MODELS = OpenAIModels()

# Completion token limits
CTOKENS = {"gpt-4o-2024-05-13": 4096, 
           "gpt-4-0125-preview": 4096, 
           "gpt-3.5-turbo-0125": 4096}

# class to access the OpenAI embeddings
class OpenAIEmbeddings:
    small = "text-embedding-3-small"
    large = "text-embedding-3-large"
    ada = "text-embedding-ada-002"
EMBEDDINGS = OpenAIEmbeddings()

# class for gettimg embeddings
class Embedder():
    def __init__(self, model=EMBEDDINGS.small):
        self.client = OpenAI()
        self.model = model

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model=self.model).data[0].embedding