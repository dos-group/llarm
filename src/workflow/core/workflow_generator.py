from openai import OpenAI
from openai import AzureOpenAI

class WorkflowGenerator:
    """
    The `WorkflowGenerator` transforms a prompt into executable Python 3 code,
    abstracting over externally hosted Large Language Model (LLM) services.
    """
    def __init__(
        self,
        prompt=None,
        model_name=None,
        model_url= None,
        model_key= None,
        model_temperature=None,
        azure_client=False,
    ):
        if azure_client is True:
            options = {}

            if model_url is not None:
                options["azure_endpoint"] = model_url

            if model_key is not None:
                options["api_key"] = model_key

            self.__client = AzureOpenAI(
                api_version="2024-07-01-preview",
                **options,
            )
        else:
            options = {}

            if model_url is not None:
                options["base_url"] = model_url

            if model_key is not None:
                options["api_key"] = model_key

            self.__client = OpenAI(**options)

        self.__prompt = prompt
        self.__model_name = model_name
        self.__model_temperature = model_temperature

    def generate(self, **kwargs):
        """
        Generates an output and extracts the relevant parts by querying a
        Large Language Model (LLM) with the provided input.

        The prompt—containing the user’s intention, the world description, and the
        available functions—is passed as input to the model. The output consists
        of Python 3 code enclosed within Markdown-style code fences, such as
        ```python```, ```python3```, or plain triple backticks (```).

        The function extracts the enclosed code and returns it.

        Example output from the LLM:
            This is the code addressing the given intention:

            python3```
            def foo():
                bar()
            ```
        """
        options = {}

        if self.__model_temperature is not None:
            options["temperature"] = self.__model_temperature

        completions = self.__client.chat.completions.create(
            model=self.__model_name,
            messages=[
                {"role": "system", "content": "You are a Python 3 code generator."},
                {
                    "role": "user",
                    "content": self.__prompt.format(**kwargs),
                }
            ],
            **options
        )

        output = completions.choices[0].message.content.strip("\n\r ")

        print(output)

        marker = "```"

        left_markers = [marker + "python3", marker + "python", marker]
        right_marker = marker

        output = output.strip("\n\r ")

        start = None
        start_padding = None

        for left_marker in left_markers:
            index = output.find(left_marker)

            if index < 0:
                continue

            start = index
            start_padding = len(left_marker)

            break

        if start is None:
            return None

        end = output.find(right_marker, start + start_padding)
        if end < 0:
            end = len(output)

        if end is None:
            return None

        return output[start + start_padding:end]
