import argparse
from measles import Bumper
from bumper import bump
from bumper import paths
from bumper.openai import MODELS

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=1, help="Number of iterations")

    # Parse the arguments
    args = parser.parse_args()

    # run Bumper
    answers = []
    model = MODELS.gpt4
    bumper = Bumper(model=model)
    for i in range(args.iter):
        bump(bumper, "When is the low transmission season in Chad?")
        bump(bumper, "What year should Chad run its next SIA?", new_thread=False)
        messages = bumper.get_messages()
        answers.append(
            f"{messages.data[4].content[0].text.value} // {messages.data[3].content[0].text.value} \n {messages.data[1].content[0].text.value} // {messages.data[0].content[0].text.value}"
        )
    bumper.cleanup()

    # write the list of string to paths.results / "figure_2.txt"
    with open(paths.results / "figure_3.txt", "w") as f:
        for item in answers:
            f.write(f"{item}\n")
