####################################################################################################

####################################################################################################

####################################################################################################

####################################################################################################

def print_separator(newlines = 0):
    columns = 80
    try:
        columns = get_terminal_size().columns
    except:
        pass
    finally:
        pass

    print(str(columns * "#") + str(newlines * "\n"))

class TestCase:
    def __init__(
            self,
            function_table,
            url,
            key,
            model,
            temperature,
            prompt,
    ):
        self.__function_table = function_table
        self.__url = url
        self.__key = key
        self.__model = model
        self.__temperature = temperature
        self.__prompt = prompt

    @property
    def url(self):
        return self.__url

    @property
    def key(self):
        return self.__key

    @property
    def model(self):
        return self.__model

    @property
    def temperature(self):
        return self.__temperature

    def run(self, context = None):
        if context is None:
            context = {}

        context["get_test_case"] = lambda: self

        print_separator(0)
        print(table.format_prompt_specification())
        print_separator(0)

        request_headers = {}

        if self.__key is not None and len(self.__key) > 0:
            request_headers["Authorization"] = "Bearer " + self.__key

        request_body={
            "model": self.__model,
            "messages": [
                {'role': 'system', 'content': 'You are a Python 3 code generator.'},
                {"role": "user", "content": format_content(self.__prompt)},
            ],
            "stream": True,
        }

        if self.__temperature is not None:
            request_body["temperature"] = float(self.__temperature)

        start_time = time.time_ns()
        time_to_first_token_ns = None

        print(self.__url, self.__model, self.__temperature, self.__prompt)
        print_separator(0)

        result = request_chat_completions(
            self.__url,
            self.__model,
            'You are a Python 3 code generator.',
            self.__key,
            format_content(self.__prompt),
            self.__temperature,
        )

        output = result["output"]
        elapsed_s = result["elapsed_s"]
        time_to_first_token_ms = result["time_to_first_token_ms"]

        print(output)
        print_separator(0)

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
            return {
                "status": "failed",
                "execution": {},
                "output": output,
                "generated_code": "",
                "response_time_in_seconds": elapsed_s,
                "time_to_first_token_in_milliseconds": time_to_first_token_ms,
            }

        end = output.find(right_marker, start + start_padding)
        if end < 0:
            end = len(output)

        if end is None:
            return {
                "status": "failed",
                "execution": {},
                "output": output,
                "generated_code": "",
                "response_time_in_seconds": elapsed_s,
                "time_to_first_token_in_milliseconds": time_to_first_token_ms,
            }

        code = output[start + start_padding:end]

        print(code)
        print_separator(0)

        try:
            execution_result = self.__function_table.evaluate(
                code,
                context = context,
                tracing = True,
            )

            print_separator(0)

            print("Success (total {}s, ttft {}ms)".format(elapsed_s, time_to_first_token_ms))

            return {
                "status": "success",
                "execution": execution_result,
                "output": output,
                "generated_code": code,
                "response_time_in_seconds": elapsed_s,
                "time_to_first_token_in_milliseconds": time_to_first_token_ms,
            }
        except Exception as e:
            print("Failed (total {}s, ttft {}ms)".format(elapsed_s, time_to_first_token_ms))
            print(format_exc())

            return {
                "status": "error",
                "execution": e,
                "output": output,
                "generated_code": code,
                "response_time_in_seconds": elapsed_s,
                "time_to_first_token_in_milliseconds": time_to_first_token_ms,
            }

        print_separator(0)

####################################################################################################

def find_file(context, expression: 'String') -> 'String|null':
    directory = "files"

    for file_name in listdir(directory):
        if not path.isfile(path.join(directory, file_name)):
            continue

        if expression not in file_name:
            continue

        return file_name

    return None

def find_all_audio_files() -> Collection['String']:
    directory = "files"

    return [
        f for f in listdir(directory) if path.isfile(path.join(directory, f))
    ]

def play_audio_file(file_path: 'String') -> None:
    result = subprocess.run(
        ["termux-media-player", "play", path.join("files", file_path)],
        text=True,
        check=True,
        capture_output=True,
    )

    print(result.stdout)

def stop_audio_player() -> None:
    result = subprocess.run(
        ["termux-media-player", "stop"],
        text=True,
        check=True,
        capture_output=True,
    )

    print(result.stdout)

def sleep(seconds: 'Integer') -> None:
    time.sleep(seconds)

def generate_random_number(context, inclusiveStart: 'Integer', exclusiveEnd: 'Integer') -> 'Integer':
    if 'seed' in context:
        seed(context['seed'])

    return randint(inclusiveStart, exclusiveEnd)

def query_llm(context, query: 'String') -> 'String':
    test_case = context["get_test_case"]()

    return request_chat_completions(
        test_case.url,
        test_case.model,
        None,
        test_case.key,
        query,
        test_case.temperature,
    )["output"]

def http_get_request(
    url: 'String',
    headers: 'Dictionary<String, String>',
) -> 'String':
    return requests.request('GET', url, headers = headers).text

def shell(
    context,
    command: 'String',
) -> 'String':
    return_value = ""
    if "shell_return_value" in context:
        return_value = context["shell_return_value"]

    Function.stub('shell', command)()

    return return_value

table = FunctionTable(FunctionSignatureFormatter())

table.register(
    Function.stub("find_contact_id", 1),
    name = "find_contact_id",
    argument_types = {
        "expression": "String",
    },
    return_type = "Integer|null",
)
table.register(
    Function.stub("find_contact_email", "john.doe@example.com"),
    name = "find_contact_email",
    argument_types = {
        "contact_id": "Integer",
    },
    return_type = "String|null",
)
table.register(
    Function.stub("ask_question", "Hello"),
    name = "ask_question",
    argument_types = {
        "question": "String",
    },
    return_type = "String",
)
table.register(
    Function.stub("send_email"),
    name = "send_email",
    argument_types = {
        "email": "String",
        "subject": "String",
        "text": "String",
        "attachment_paths": "Collection<String>",
    },
    return_type = None,
)
table.register(
    Function.stub("get_temperature", 37),
    name = "get_temperature",
    argument_types = {},
    return_type = "Integer",
)
table.register(
    Function.stub("find_files", [
        "File0",
        "File1",
        "File2",
        "File3",
        "File4",
    ]),
    name = "find_files",
    argument_types = {
        "expression": "String",
    },
    return_type = "Collection<String>"
)
table.register(
    Function.stub("print"),
    name = "print",
    argument_types = {
        "text": "String",
    },
)
table.register(shell)
table.register(sleep)
table.register(find_all_audio_files)
table.register(generate_random_number)
table.register(play_audio_file)
table.register(find_file)
table.register(stop_audio_player)
table.register(query_llm)
table.register(http_get_request)

####################################################################################################

class Encoder(JSONEncoder):
    def default(self, instance):
        if isinstance(instance, Exception):
            return str(instance)

        if callable(instance):
            return {}

        return super().default(instance)

####################################################################################################

intentions = [
    {
        "prompt": "Please sleep for 5 seconds",
        "context" : {},
    },
    ################################################################################################
    {
        "prompt": "Please tell me a random number between 1 and 100",
        "context": {},
    },
    ################################################################################################
    {
        "prompt": "Please tell me the current temperature",
        "context": {},
    },
    ################################################################################################
    {
        "prompt": "Play a random song in my list for 5 seconds",
        "context" : {},
    },
    ################################################################################################
    {
        "prompt": "Which is the largest city in germany?",
        "context": {},
    },
    ################################################################################################
    {
        "prompt": "Please tell me all files in the current directory",
        "context" : {
            "shell_return_value": "File0\nFile1\nFile2"
        },
    },
    ################################################################################################
    {
        "prompt": "Please send my car title to my insurance company",
        "context" : {},
    },
    ################################################################################################
    {
        "prompt": "Please summarize the wikipedia article https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        "context" : {},
    },
    ################################################################################################
    {
        "prompt": "Please install nginx on the machine with the address 127.0.0.1:2222 running Debian GNU/Linux",
        "context" : {
            "shell_return_value": "",
        },
    },
]

####################################################################################################

DEFAULT_ENDPOINT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_MODEL_TEMPERATURE = 0.0

context = {
    "seed": 2 ** 64 - 1,
}

if len(argv) > 2 and argv[1] == 'experiments':
    for index, intention in enumerate(intentions):
        test_case = TestCase(
            table,
            environ.get("ENDPOINT_URL", DEFAULT_ENDPOINT_URL),
            environ.get("ENDPOINT_KEY", None),
            environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
            environ.get("MODEL_TEMPERATURE", DEFAULT_MODEL_TEMPERATURE),
            intention["prompt"],
        )

        llm_result = test_case.run(context)

        output_path = "outputs/{}".format(test_case.model)
        output_file = "{}/{}.json".format(output_path, index)
        code_file = "{}/{}.py".format(output_path, index)

        if not path.exists(output_path):
            makedirs(output_path)

        with open(output_file, "w") as f:
            f.write(
                dumps(
                    {
                        **llm_result,
                        "prompt": intention["prompt"],
                    },
                    cls = Encoder,
                    indent = 4
                )
            )

        with open(code_file, "w") as f:
            f.write(llm_result["generated_code"])
else:
    test_case = TestCase(
        table,
        environ.get("ENDPOINT_URL", DEFAULT_ENDPOINT_URL),
        environ.get("ENDPOINT_KEY", None),
        environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
        environ.get("MODEL_TEMPERATURE", DEFAULT_MODEL_TEMPERATURE),
        stdin.read(),
    )

    llm_result = test_case.run(context)

    print(llm_result)
