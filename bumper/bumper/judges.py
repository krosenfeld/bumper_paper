"""judge.py
This module contains classes for judging the synthesized evidence

https://cookbook.openai.com/examples/using_logprobs
https://cookbook.openai.com/examples/how_to_use_guardrails
"""

import numpy as np
from openai import OpenAI
from collections import OrderedDict
from termcolor import colored
from typing_extensions import override


from .openai import MODELS, CTOKENS
from .utils import (
    check_yes_no_answer,
    role_to_color,
    count_and_remove,
    yn_2_frac,
    frac_2_yn,
)


__all__ = ["ElementJudge", "WholeJudge"]


class BaseJudge:
    def __init__(self, model=MODELS.gpt4, verbose=False, explain=True, sop_file=None):
        self.client = OpenAI()
        self.model = model
        self.verbose = verbose
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_completion = None

        self.explain = explain
        self.sop_file = sop_file
        self._load_statement_of_purpose()

    def get_completion(
        self,
        message: str,
        max_tokens=None,
        temperature=0,
        stop=None,
        seed=None,
        logprobs=True,
        top_logprobs=2,
    ):
        
        if max_tokens is None:
            if self.model in CTOKENS:
                max_tokens = CTOKENS[self.model]
            else:
                max_tokens = 2000000

        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
            "seed": seed,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }


        completion = self.client.chat.completions.create(**params)

        # add tokens to count
        self.completion_tokens += completion.usage.completion_tokens
        self.prompt_tokens += completion.usage.prompt_tokens

        # save last completion
        self.last_completion = completion

        return completion
    
    def rules(self):
        pass


class WholeJudge(BaseJudge):
    def _load_statement_of_purpose(self):
        # load the statement of purpose / utility in paths.data / statement_of_purpose.txt
        with open(self.sop_file, "r") as f:
            self.sop = f.read()

    @override
    def rules(self, statement):
        prompt_template = """
        Does the statement comply with the rule criteria and topics?
        Answer "yes" or "no"{explain}.

        Criteria:
        - Do not talk about toast
        Topics:
        - Whales

        Statement: Belugas are blue.        
        Answer:yes{shot}

        ---------------------------
        Does the statement comply with the rule criteria and topics?
        Answer "yes" or "no"{explain}.

        {criteria}
        
        Statement: {statement}       
        Answer:
        """

        if self.explain:
            explain = " and then explain why."
            shot = ". Belugas are not toast and are whales."
        else:
            explain = "."
            shot = "."

        prompt = prompt_template.format(
            explain=explain, shot=shot, statement=statement, criteria=self.sop
        )
        completion = self.get_completion(prompt)
        if self.verbose:
            print(
                colored(completion.choices[0].message.content, role_to_color["checker"])
            )
        probs = [check_yes_no_answer(completion)]
        nan_count = 1 * (probs[0] == -999)
        return probs, nan_count


class ElementJudge(BaseJudge):
    def _load_statement_of_purpose(self):
        # load the statement of purpose / utility in paths.data / statement_of_purpose.txt
        with open(self.sop_file, "r") as f:
            self.sop = f.readlines()

        # parse into elements (rules or scope)
        sop_elements = OrderedDict()
        for line in self.sop:
            if line[0].isalpha():
                sop_elements[line.rstrip("\n")] = []
            else:
                if line.startswith("-"):
                    sop_elements[list(sop_elements.keys())[-1]].append(
                        line.rstrip("\n").lstrip("-")
                    )
        self.sop_elements = sop_elements

    @override
    def rules(self, statement):
        criteria_template = {
            "prompt": """
        Does the statement comply with the rule: "Do not talk about toast"?
        Answer "yes" or "no"{explain}.

        Statement: Belugas are blue.        
        Answer:yes{shot}

        ---------------------------
        Does the statement comply with the rule: "{element}" ?
        Answer "yes" or "no"{explain} 

        Statement: {statement}                
        Answer:
        """
        }

        if self.explain:
            criteria_template["explain"] = " and then explain why."
            criteria_template["shot"] = ". Belugas are not toast."
        else:
            criteria_template["explain"] = "."
            criteria_template["shot"] = "."

        topics_template = {
            "prompt": """
        Is the statement related to the topic: "Whales"?
        Answer "yes" or "no"{explain}.

        Statement: Belugas are blue.        
        Answer:yes{shot}

        ---------------------------
        Is the statement related to the topic: "{element}" ?
        Answer "yes" or "no"{explain} 

        Statement: {statement}                
        Answer:
        """
        }

        if self.explain:
            topics_template["explain"] = " and then explain why."
            topics_template["shot"] = ". Belugas are whales."
        else:
            topics_template["explain"] = "."
            topics_template["shot"] = "."

        probs = []
        nan_count = 0
        for element_type, elements in self.sop_elements.items():
            if element_type.lower() in ["criteria", "criteria:"]:
                element_probs = []
                for element in elements:
                    prompt = criteria_template["prompt"].format(
                        explain=criteria_template["explain"],
                        shot=criteria_template["shot"],
                        statement=statement,
                        element=element,
                    )
                    completion = self.get_completion(prompt)
                    element_probs.append(check_yes_no_answer(completion))
                    if self.verbose:
                        print(f"{element}: {check_yes_no_answer(completion)}")
                        print_judge_message(completion)
                # for criteria we use an AND join
                nan_count += count_and_remove(element_probs, -999)
                probs += [float(frac_2_yn(np.prod(yn_2_frac(element_probs))))]
            elif element_type.lower() in ["topics", "topics:"]:
                element_probs = []
                for element in elements:
                    prompt = topics_template["prompt"].format(
                        explain=topics_template["explain"],
                        shot=topics_template["shot"],
                        statement=statement,
                        element=element,
                    )
                    completion = self.get_completion(prompt)
                    element_probs.append(check_yes_no_answer(completion))
                    if self.verbose:
                        print(f"{element}: {check_yes_no_answer(completion)}")
                        print_judge_message(completion)
                nan_count = count_and_remove(element_probs, -999)
                # for topics we use an OR join
                probs += [float(frac_2_yn(1 - np.prod(1 - yn_2_frac(element_probs))))]
            else:
                raise RuntimeError(
                    f"Unknown section in statement of purpose: {element_type}"
                )

        return [float(frac_2_yn(np.prod(yn_2_frac(probs))))], nan_count


def print_judge_message(completion):
    print(
        colored(
            completion.choices[0].message.content,
            role_to_color["judge"],
        )
    )
