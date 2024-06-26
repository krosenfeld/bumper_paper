from pathlib import Path
from bumper import BaseBumper
from typing_extensions import override

from .tools import tool_specs, tool_functions

class Bumper(BaseBumper):

    @override
    def __init__(
        self,
        name="RugbyBumper",
        sop_file=None,
        **kwargs
    ):
        if sop_file is None:
            sop_file = Path(__file__).resolve().parent / "guidelines.txt"
        super().__init__(name=name, sop_file=sop_file, **kwargs)
 
        # add tools
        self.tools = tool_specs

        # gather all available functions
        self.available_functions = tool_functions

        # create the assistant
        self._create()
        self.create_thread()            

    @override
    def launch_tools(self):
        """Launch the tools available to the assistant"""

        # Define the list to store tool outputs
        tool_outputs = []

        # Loop through each tool in the required action section
        for tool in self.run.required_action.submit_tool_outputs.tool_calls:
            function_name = tool.function.name
            print(f"calling {function_name}")
            function_to_call = self.available_functions[function_name]
            # function_args = json.loads(tool.function.arguments)        
            if function_name in [
                "get_defense_stats",
                "get_attack_stats"
            ]:
                function_response = function_to_call()
                tool_outputs.append(
                    {"tool_call_id": tool.id, "output": function_response}
                )                
            else:
                raise RuntimeError(
                    f"Function {function_name} not found in available functions"
                )

        # Submit all tool outputs at once after collecting them in a list
        if tool_outputs:
            try:
                self.run = self.client.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=self.thread.id,
                    run_id=self.run.id,
                    tool_outputs=tool_outputs,
                )
                print("Tool outputs submitted successfully.")
            except Exception as e:
                print("Failed to submit tool outputs:", e)
        else:
            print("No tool outputs to submit.")

        return tool_outputs                