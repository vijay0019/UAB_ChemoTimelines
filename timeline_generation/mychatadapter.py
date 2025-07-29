import json_repair
import re
import textwrap
from typing import Any, Dict, NamedTuple, Optional, Type, get_args, get_origin

from litellm import ContextWindowExceededError
from pydantic.fields import FieldInfo

import dspy
from dspy.adapters.base import Adapter
from dspy.adapters.utils import *
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError

from typing import TYPE_CHECKING, Any, Optional, Type

from dspy.adapters.types import History
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")


class Date(NamedTuple):
	year: int
	month: int|None
	day_of_month: int|None
	week_of_year: int|None


def parse_value(value, annotation):
    if annotation is str:
        return str(value)

    if isinstance(annotation, enum.EnumMeta):
        return find_enum_member(annotation, value)

    origin = get_origin(annotation)

    if origin is Literal:
        allowed = get_args(annotation)
        if value in allowed:
            return value

        if isinstance(value, str):
            v = value.strip()
            if v.startswith(("Literal[", "str[")) and v.endswith("]"):
                v = v[v.find("[") + 1 : -1]
            if len(v) > 1 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]

            if v in allowed:
                return v

        raise ValueError(f"{value!r} is not one of {allowed!r}")

    if not isinstance(value, str):
        return TypeAdapter(annotation).validate_python(value, strict=False)

    candidate = ""#json_repair.loads(value)  # json_repair.loads returns "" on failure.
    if candidate == "" and value != "":
        try:
            # candidate = ast.literal_eval(value)
            candidate = eval(value)
        except (ValueError, SyntaxError):
            candidate = value

    try:
        return TypeAdapter(annotation).validate_python(candidate, strict=False)
    except pydantic.ValidationError:
        if origin is Union and type(None) in get_args(annotation) and str in get_args(annotation):
            return str(candidate)
        raise


class MyChatAdapter(dspy.ChatAdapter):
    def parse(self, signature: Type[Signature], completion: str) -> dict[str, Any]:
        print(f"Completion before postprocessing: {completion}")
        reasoning = re.search(r"<think>(.*?)</think>", completion, flags=re.DOTALL)
        if reasoning:
            reasoning = reasoning.group(1).strip()
            completion = completion.replace(reasoning, "").replace("<think>", "").replace("</think>", "")
        else:
            reasoning = ""
        sections = [(None, [])]
        completion = re.sub(r"Literal\['finish'\]", r"finish", completion)
        completion = re.sub(r"Literal(\[.*?\])", r"\1", completion)
        completion = re.sub(r"```(?:python|json|markdown|plaintext)\n(.*?)\n```", r"\1", completion, flags=re.DOTALL)
        completion = re.sub(r"```(?:python|json|markdown|plaintext)?", "", completion, flags=re.DOTALL)
        completion = completion.replace("python", "")
        if re.search(r"\[\[ \n##", completion):
            completion = re.sub(r"\[\[ \n## (\w+) ##(?: \]\])?", r"[[ ## \1 ## ]]", completion)
            completion = re.sub(r"\n\]\]\n", "\n", completion)
        if re.search(r"\[\[ ## \w+ ##\n", completion):
            completion = re.sub(r"\[\[ ## (\w+) ##\n", r"[[ ## \1 ## ]]\n", completion)
            completion = re.sub(r"\n\]\]\n", "\n", completion)
        completion = re.sub(r"\w+ = \{", "{", completion)
        completion = re.sub(r"\[\s+\[", "[", completion)
        completion = re.sub(r"\]\s+\]", "]", completion)
        completion = re.sub(r"\],\s+\[", "", completion)
        completion = completion.replace("[[[assistant", "[[ ## completed ## ]]").replace("[[end]]", "[[ ## completed ## ]]").replace("[[[<paste>]]]", "[[ ## completed ## ]]")
        completion = re.sub(r"\}(?!.*\}).*?$", "}\n\n[[ ## completed ## ]]", completion, flags=re.DOTALL)
        completion = completion.replace("### [[ ##", "[[ ##")
        completion = re.sub(r"### Timeline:?", "[[ ## Timeline ## ]]", completion)
        completion = re.sub(r"### Reasoning:?", "[[ ## reasoning ## ]]", completion)
        completion = completion.replace("]]\n\n", "]]\n")
        completion = re.sub('"True"', "True", completion, flags=re.IGNORECASE)
        completion = re.sub('"False"', "False", completion, flags=re.IGNORECASE)
        missing_field = []
        if len(signature.output_fields) == 1:
            field = list(signature.output_fields.keys())[0]
            if field not in completion:
                completion = re.sub(r"(.*)(\{.*?\}\s+\[\[ ## completed ## \]\])", fr"\1\n\n[[ ## {field} ## ]]\n\n\2", completion, flags=re.DOTALL)
        for field in signature.output_fields:
            if "[[ ## " + field + " ## ]]" not in completion:
                missing_field.append(field)
                if "## " + field + " ##" in completion:
                    completion = re.sub(r"## " + field + r" ##", r"[[ ## " + field + r" ## ]]", completion)
                    if "[[ ## completed ## ]]" not in completion:
                        completion = re.sub(r"## completed ##", r"[[ ## completed ## ]]", completion)
                    continue
                completion = re.sub(r"\[\[ ## " + field + r" ## \]\](.*?\[\[ ## completed ## \]\])", r"[[ ## " + field + r" ## ]]\1", completion, flags=re.DOTALL)
            if "[[ ## " + field + " ## ]]" in completion and field in missing_field:
                missing_field.remove(field)
        if len(missing_field) == 1:
            completion = re.sub(r"\[\[ ## completed ## \]\](.*?\[\[ ## completed ## \]\])", r"[[ ## " + missing_field[0] + r" ## ]]\1", completion, flags=re.DOTALL)
            if "[[ ## " + missing_field[0] + " ## ]]" in completion:
                missing_field = []
        completion = re.sub(r"\[{3,} ## (\w+) ## \]{3,}", r"[[ ## \1 ## ]]", completion)
        completion = re.sub(r"'(.*?[a-z]'s.*?)'", r'"\1"', completion)
        completion = completion.replace(")\n\n[[ ## completed ## ]]", ")}\n\n[[ ## completed ## ]]")
        completion = re.sub(r"### (\[\[ ## \w+ ## \]\])", r"\1", completion)
        completion = completion.replace("### [[## Reasoning ##]]", "[[ ## reasoning ## ]]")
        ### NEW FOR COMPETITION
        completion = re.sub("null", "None", completion, flags=re.IGNORECASE)
        if "next_tool_name" in completion:
            completion = re.sub(r"(\[\[ ## next_tool_name ## \]\]\s+)[^a-z0-9_\n]+([a-z0-9_]+)[^a-z0-9_\n]+", r"\1\2", completion)
        else:
            # doesn't use Date object
            completion = re.sub(r"\[(\d+), (\d+|None), (\d+|None), (\d+|None)\]", r"Date(year=\1, month=\2, day_of_month=\3, week_of_year=\4)", completion)
            completion = re.sub(r"\[(\d+), (\d+), (\d+)\]", r"Date(year=\1, month=\2, day_of_month=\3, week_of_year=None)", completion)
            completion = re.sub(r"\[(\d+), (\d+)\]", r"Date(year=\1, month=\2, day_of_month=None, week_of_year=None)", completion)
            completion = re.sub(r"\[(\d+)\]", r"Date(year=\1, month=None, day_of_month=None, week_of_year=None)", completion)
        if "Date(" in completion:
            # forgets to include all components of Date
            completion = re.sub(r"Date\(year=(\d+)\)\)", r"Date(year=\1, month=None, day_of_month=None, week_of_year=None))", completion)
            completion = re.sub(r"month=(\d+)\)\)", r"month=\1, day_of_month=None, week_of_year=None))", completion)
            completion = re.sub(r"day_of_month=(\d+)\)\)", r"day_of_month=\1, week_of_year=None))", completion)
            # forgets to name components of Date
            completion = re.sub(r"Date\((\d+), (\d+|None), (\d+|None), (\d+|None)\)", r"Date(year=\1, month=\2, day_of_month=\3, week_of_year=\4)", completion)
            completion = re.sub(r"Date\((\d+), (\d+), (\d+)\)", r"Date(year=\1, month=\2, day_of_month=\3, week_of_year=None)", completion)
            completion = re.sub(r"Date\((\d+), (\d+)\)", r"Date(year=\1, month=\2, day_of_month=None, week_of_year=None)", completion)
            completion = re.sub(r"Date\((\d+)\)", r"Date(year=\1, month=None, day_of_month=None, week_of_year=None)", completion)
            # ? not sure why this is happening
            completion = re.sub(r"(day_of_month=\d+), day_of_month=None", r"\1", completion)
            completion = re.sub(r"Date\(year=None.*", "", completion)
        # adds extraneous text after list
        completion = re.sub(r"(\[\[ ## remove ## \]\]\n\[.*?\]).*", r"\1\n\n[[ ## completed ## ]]", completion, flags=re.DOTALL)
        completion = re.sub("### Step-by-step Reasoning", "[[ ## reasoning ## ]]", completion)
        completion = re.sub(r"\[\[ ## Timeline ## \]\] Update", "[[ ## timeline_update ## ]]", completion)
        print(f"Completion after postprocessing: {completion}")
        if missing_field:
            pass
            # print(
            #     f"Missing fields in the LM response: {', '.join(missing_field)}. Please check the LM response for any missing fields."
            # )
        for line in completion.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                # If the header pattern is found, split the rest of the line as content
                header = match.group(1)
                remaining_content = line[match.end() :].strip()
                sections.append((header, [remaining_content] if remaining_content else []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        fields = {"reasoning": reasoning} if reasoning else {}
        for k, v in sections:
            if (k not in fields) and (k in signature.output_fields):
                try:
                    fields[k] = parse_value(v, signature.output_fields[k].annotation)
                except Exception as e:
                    # print(f"Error parsing field '{k}': {e}")
                    fields[k] = None
        if fields.keys() != signature.output_fields.keys():
            # print(
            #     f"Missing fields in the LM response: {', '.join(set(signature.output_fields.keys()) - set(fields.keys()))}. Please check the LM response for any missing fields."
            # )
            for k in signature.output_fields:
                if k not in fields.keys():
                    fields[k] = None

        return fields

    # def __call__(
    #     self,
    #     lm: "LM",
    #     lm_kwargs: dict[str, Any],
    #     signature: Type[Signature],
    #     demos: list[dict[str, Any]],
    #     inputs: dict[str, Any],
    # ) -> list[dict[str, Any]]:
    #     processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
    #     inputs = self.format(processed_signature, demos, inputs)

    #     for item in inputs:
    #         content = item.get("content", "")
    #         if isinstance(content, str):
    #             # Remove English stop words from the content
    #             content = re.sub(
    #                 r"\b(?:{})\b".format("|".join(ENGLISH_STOP_WORDS)),
    #                 "",
    #                 content,
    #                 flags=re.IGNORECASE,
    #             ).strip()
    #             content = re.sub(r" +", " ", content)  # Remove extra spaces
    #         item["content"] = content

    #     outputs = lm(messages=inputs, **lm_kwargs)
    #     return self._call_postprocess(processed_signature, signature, outputs)

    # def format(
    #     self,
    #     signature: Type[Signature],
    #     demos: list[dict[str, Any]],
    #     inputs: dict[str, Any],
    # ) -> list[dict[str, Any]]:
    #     """Format the input messages for the LM call.

    #     This method converts the DSPy structured input along with few-shot examples and conversation history into
    #     multiturn messages as expected by the LM. For custom adapters, this method can be overridden to customize
    #     the formatting of the input messages.

    #     In general we recommend the messages to have the following structure:
    #     ```
    #     [
    #         {"role": "system", "content": system_message},
    #         # Begin few-shot examples
    #         {"role": "user", "content": few_shot_example_1_input},
    #         {"role": "assistant", "content": few_shot_example_1_output},
    #         {"role": "user", "content": few_shot_example_2_input},
    #         {"role": "assistant", "content": few_shot_example_2_output},
    #         ...
    #         # End few-shot examples
    #         # Begin conversation history
    #         {"role": "user", "content": conversation_history_1_input},
    #         {"role": "assistant", "content": conversation_history_1_output},
    #         {"role": "user", "content": conversation_history_2_input},
    #         {"role": "assistant", "content": conversation_history_2_output},
    #         ...
    #         # End conversation history
    #         {"role": "user", "content": current_input},
    #     ]

    #     And system message should contain the field description, field structure, and task description.
    #     ```


    #     Args:
    #         signature: The DSPy signature for which to format the input messages.
    #         demos: A list of few-shot examples.
    #         inputs: The input arguments to the DSPy module.

    #     Returns:
    #         A list of multiturn messages as expected by the LM.
    #     """
    #     inputs_copy = dict(inputs)

    #     # If the signature and inputs have conversation history, we need to format the conversation history and
    #     # remove the history field from the signature.
    #     history_field_name = self._get_history_field_name(signature)
    #     if history_field_name:
    #         # In order to format the conversation history, we need to remove the history field from the signature.
    #         signature_without_history = signature.delete(history_field_name)
    #         conversation_history = self.format_conversation_history(
    #             signature_without_history,
    #             history_field_name,
    #             inputs_copy,
    #         )

    #     messages = []
    #     system_message = (
    #         # "/no_think\n"
    #         f"{self.format_field_description(signature)}\n"
    #         f"{self.format_field_structure(signature)}\n"
    #         f"{self.format_task_description(signature)}\n"
    #         "Do not overthink!"
    #     )
    #     messages.append({"role": "system", "content": system_message})
    #     messages.extend(self.format_demos(signature, demos))
    #     if history_field_name:
    #         # Conversation history and current input
    #         content = self.format_user_message_content(signature_without_history, inputs_copy, main_request=True)
    #         messages.extend(conversation_history)
    #         messages.append({"role": "user", "content": content})
    #     else:
    #         # Only current input
    #         content = self.format_user_message_content(signature, inputs_copy, main_request=True)
    #         messages.append({"role": "user", "content": content})

    #     messages = try_expand_image_tags(messages)
    #     return messages
