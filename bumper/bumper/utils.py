import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored  

c8c = {
    "red": "#ED0A3F",
    "orange": "#FF8833",
    "yellow": "#FBE870",
    "green": "#01A638",
    "blue": "#0066FF",
    "violet": "#803790",
    "brown": "#AF593E",
    "black": "#000000",
}

role_to_color = {
    "system": "red",
    "user": "green",
    "bumper": "blue",
    "function": "magenta",
    "judge": "cyan"
}

def check_yes_no_answer(completion, nan_value = -999, prefix="answer: "):
    # Result for each token in the sequence
    token_logprobs = completion.choices[0].logprobs.content
    # Use the first token for logprob
    linprob = linear_prob(token_logprobs[0].top_logprobs[0].logprob)
    # Check for result (yes/no)
    msg = completion.choices[0].message.content.lower().lstrip()
    if msg.startswith(prefix):
        msg = msg[len(prefix):]
    if (msg.startswith('no')) or (msg.startswith('n0')):
        return -1*linprob
    elif msg.startswith('yes'):
        return linprob
    else:
        return nan_value


def linear_prob(logprob, ndigits=2):
    # https://cookbook.openai.com/examples/using_logprobs
    return np.exp(logprob)*100

def pretty_print_conversation(messages):
    """ Pretty print the conversation between the user, assistant, and system.
    https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
    """
    
    for message in reversed(messages.data):
        if message.role == "user":
            print(colored(f"user: {message.content[0].text.value}\n", role_to_color["user"]))
        elif message.role == "system":
            print(colored(f"system: {message.content[0].text.value}\n", role_to_color["system"]))
        else:
            print(colored(f"BUMPER: {message.content[0].text.value}\n", role_to_color["bumper"]))

def axes_setup(axes, left_visible=True):
    """
    Setup a matplotlib axis with custom settings.
    """
    axes.spines["left"].set_position(("axes", -0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_visible(left_visible)
    axes.tick_params(axis="x", which="both", labelbottom=True)
    return axes

def plot_setup(**kwargs):
    # Plot environment
    plt.rcParams["font.size"] = 22.0
    plt.rcParams["xtick.labelsize"] = "medium"
    plt.rcParams["ytick.labelsize"] = "medium"
    plt.rcParams["legend.fontsize"] = "medium"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    # set default dpi
    plt.rcParams["figure.dpi"] = 300
    for k,v in kwargs.items():
        plt.rcParams[k] = v

def set_xaxis(ax, xticks=[-1, -0.5, 0, 0.5, 1]):
    ax.set_xlim([-1.01, 1.01]) # to avoid histogram bars to be cut off
    ax.set_xticks(xticks)
    xtickl = [str(np.abs(x)) for x in xticks]
    xtickl[0] = xtickl[0] + '\nS|fail'
    xtickl[-1] = xtickl[-1] + '\nS|pass'
    ax.set_xticklabels(xtickl)

def count_and_remove(lst, val):
    """ Count the number of elements with value 'val' in a list and remove them."""
    count = 0
    while val in lst:
        lst.remove(val)
        count += 1
    return count

def yn_2_frac(yn):
    """no-yes scale to fraction: [-100, 100] -> [0, 1]"""
    return (0.5/100)*np.array(yn) + 0.5

def frac_2_yn(frac):
    """fraction to no-yes scale: [0, 1] -> [-100, 100]"""
    return 200*np.array(frac) - 100

def insert_newline_midway(text):
    midpoint = len(text) // 2  # Find the rough midpoint of the string
    
    # Find the nearest space to the midpoint
    for i in range(midpoint, len(text)):
        if text[i] == ' ':
            return text[:i] + '\n' + text[i+1:]
    for i in range(midpoint, 0, -1):
        if text[i] == ' ':
            return text[:i] + '\n' + text[i+1:]
    
    # If no space is found, we return the original text
    return text
