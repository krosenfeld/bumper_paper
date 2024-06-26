import json
from measles import Bumper
from bumper import bump
from bumper import paths
from bumper.openai import MODELS

Qs = ["Should SIAs be run earlier in the year in Burkina Faso or Ethiopia?",
      "Is it easier to run SIAs in Pakistan or Afghanistan?"]

def run_experiment(n_iter=5):
    model = MODELS.gpt4
    res = []
    for Q in Qs:
        bumper = Bumper(model=model)

        answers = []
        for i in range(n_iter):
            bump(bumper, Q, n=1)
            messages = bumper.get_messages()
            res.append({Q:
                f"{messages.data[1].content[0].text.value} // {messages.data[0].content[0].text.value}"}
            )

    bumper.cleanup()

    # Open a file in write mode
    with open(paths.results / "table_2.jsonl", 'w') as file:
        for item in res:
            # Convert dictionary to JSON string
            json_string = json.dumps(item, indent=4, sort_keys=True)
            # Write JSON string to file with a newline character
            file.write(json_string + '\n')

if __name__ == "__main__":
    run_experiment()