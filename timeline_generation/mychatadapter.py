import json_repair
import re
import textwrap
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Type as TypingType, get_args, get_origin

from litellm import ContextWindowExceededError
from pydantic.fields import FieldInfo

import dspy
from dspy.adapters.base import Adapter
from dspy.adapters.utils import *
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError

from dspy.adapters.types import History
from dspy.utils.callback import BaseCallback, with_callbacks

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")

CHEMO_DRUGS = ["a.c", "a/c", "abraxane", "ac", "adriamycin", "aflibercept", "albumin-bound paclitaxel",
               "alfa-2b interferon", "alibercept", "alpha-2b interferon", "anastrozole", "arimidex",
               "avastin", "bevacizumab", "caboplatin", "cabotaxol", "carbo", "carboplatin", "carbotaxol",
               "chemo", "chemotherapy", "cisplatin", "cistoplatin", "cyclophosphamide", "cytoxan",
               "docetaxel", "docetaxol", "doxil", "doxorubicin", "faslodex", "femara", "gemcitabine",
               "gemzar", "herceptin", "il-2", "il2", "interferon", "interleukin-2", "ipilimumab", "kadcyla",
               "liposomal doxorubicin", "methotrexate", "paclitaxel", "tamoxifen", "tax", "taxol",
               "taxotere", "tc", "tch", "temozolomide", "vaccinia", "vaccinia virus", "zometa", "zelboraf"]

Relation = Literal[
    "begins-on",
    "ends-on",
    "contains-1"
]


class Date(NamedTuple):
    year: int
    month: int | None = None
    day_of_month: int | None = None
    week_of_year: int | None = None

    @classmethod
    def from_partial(cls, data):
        """Create Date from partial data, handling various input formats."""
        if isinstance(data, (list, tuple)):
            # Handle [year, month] or [year, month, day] formats
            if len(data) >= 1:
                year = data[0]
                month = data[1] if len(data) > 1 and data[1] is not None else None
                day_of_month = data[2] if len(data) > 2 and data[2] is not None else None
                week_of_year = data[3] if len(data) > 3 and data[3] is not None else None
                return cls(year=year, month=month, day_of_month=day_of_month, week_of_year=week_of_year)
        elif isinstance(data, dict):
            # Handle dictionary format
            return cls(
                year=data.get('year'),
                month=data.get('month'),
                day_of_month=data.get('day_of_month'),
                week_of_year=data.get('week_of_year')
            )
        elif isinstance(data, int):
            # Handle year-only format
            return cls(year=data)
        
        # Fallback to default construction
        if hasattr(data, 'year'):
            return cls(
                year=data.year,
                month=getattr(data, 'month', None),
                day_of_month=getattr(data, 'day_of_month', None),
                week_of_year=getattr(data, 'week_of_year', None)
            )
        
        raise ValueError(f"Cannot create Date from {data}")
    
    def validate(self):
        """Validate the Date object according to TIMEX standards."""
        if self.year is None:
            raise ValueError("Year is required")
        
        if self.month is not None and not (1 <= self.month <= 12):
            raise ValueError(f"Month must be between 1-12, got {self.month}")
            
        if self.day_of_month is not None and not (1 <= self.day_of_month <= 31):
            raise ValueError(f"Day must be between 1-31, got {self.day_of_month}")
            
        if self.week_of_year is not None and not (1 <= self.week_of_year <= 53):
            raise ValueError(f"Week must be between 1-53, got {self.week_of_year}")
        
        # TIMEX constraint: if week is specified, month and day should typically be None
        if self.week_of_year is not None and self.day_of_month is not None:
            # Allow it but prioritize day_of_month for formatting
            pass
            
        return self


# Timeline = list[tuple[Literal[*CHEMO_DRUGS], Relation, Date]]
Timeline = list[tuple[str, Relation, Date]]


class Update(NamedTuple):
    add: Timeline
    remove: Timeline


def parse_value(value, annotation):
    if annotation is str:
        return str(value)

    if isinstance(annotation, enum.EnumMeta):
        return find_enum_member(annotation, value)

    # Special handling for Date class
    if annotation == Date or (hasattr(annotation, '__name__') and annotation.__name__ == 'Date'):
        try:
            if isinstance(value, str):
                # Try to evaluate string representation
                candidate = eval(value)
            else:
                candidate = value
                
            # Use the new from_partial class method
            date_obj = Date.from_partial(candidate)
            return date_obj.validate()
        except Exception as e:
            print(f"Error parsing Date from {value}: {e}")
            # Fallback: try to extract at least the year
            if isinstance(value, (list, tuple)) and len(value) > 0:
                year = value[0] if value[0] is not None else 2000  # fallback year
                return Date(year=year)
            elif isinstance(value, str):
                # Try to extract year from string
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', value)
                if year_match:
                    return Date(year=int(year_match.group()))
            # Last resort fallback
            return Date(year=2000)

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
    def parse(self, signature: TypingType[Signature], completion: str) -> dict[str, Any]:
        # print(f"Completion before postprocessing: {completion}")
        thinking = re.search(r"<think>(.*?)</think>", completion, flags=re.DOTALL)
        if thinking:
            thinking = thinking.group(1).strip()
            completion = completion.replace(thinking, "").replace("<think>", "").replace("</think>", "")
        else:
            thinking = ""
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
        # completion = re.sub(r"\[\s+\[", "[", completion)
        # completion = re.sub(r"\]\s+\]", "]", completion)
        # completion = re.sub(r"\],\s+\[", "", completion)  # oops!
        completion = completion.replace("[[[assistant", "[[ ## completed ## ]]").replace("[[end]]", "[[ ## completed ## ]]").replace("[[[<paste>]]]", "[[ ## completed ## ]]")
        completion = re.sub(r"\}(?!.*\}).*?$", "}\n\n[[ ## completed ## ]]", completion, flags=re.DOTALL)
        completion = completion.replace("### [[ ##", "[[ ##")
        completion = re.sub(r"### Timeline:?", "[[ ## Timeline ## ]]", completion)
        completion = re.sub(r"### Reasoning:?$", "[[ ## reasoning ## ]]", completion)
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
        completion = completion.replace("Update([])", "Update(add=[], remove=[])")
        completion = re.sub(r"(\[\[ ## timeline_update ## \]\]\n).*?(Update\(\s*add=\[.*?\],\s*remove=\[.*?\]\s*\)).*", r"\1\2\n\n[[ ## completed ## ]]", completion, flags=re.DOTALL)
        if "[[ ## reasoning ## ]]" not in completion:
            completion = re.sub("### Step-by-step Reasoning", "[[ ## reasoning ## ]]", completion)
        completion = re.sub(r"\[\[ ## Timeline ## \]\] Update", "[[ ## timeline_update ## ]]", completion)
        if "next_tool_name" not in completion:
            completion = completion.replace(")}\n\n[[ ## completed ## ]]", ")\n\n[[ ## completed ## ]]")
        # print(f"Completion after postprocessing: {completion}")
        if missing_field:
            print(
                f"Missing fields in the LM response: {', '.join(missing_field)}. Please check the LM response for any missing fields."
            )
            # Add retry suggestion for critical missing fields
            if any(field in ['timeline', 'cleaned_timeline'] for field in missing_field):
                print("SUGGESTION: Consider retrying with a more explicit prompt or increasing temperature.")
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

        fields = {"thinking": thinking} if thinking else {}
        for k, v in sections:
            if (k not in fields) and (k in signature.output_fields):
                try:
                    fields[k] = parse_value(v, signature.output_fields[k].annotation)
                except Exception as e:
                    print(f"Error parsing field '{k}': {e}")
                    fields[k] = None
        if fields.keys() != signature.output_fields.keys():
            missing_fields = set(signature.output_fields.keys()) - set(fields.keys())
            print(
                f"Missing fields in the LM response: {', '.join(missing_fields)}. Please check the LM response for any missing fields."
            )
            # Add retry suggestion for critical missing fields
            if any(field in ['timeline', 'cleaned_timeline'] for field in missing_fields):
                print("SUGGESTION: Consider retrying with a more explicit prompt or increasing temperature.")
                
            for k in signature.output_fields:
                if k not in fields.keys():
                    # Provide sensible defaults for known field types
                    if k in ['timeline', 'cleaned_timeline']:
                        fields[k] = []  # Empty timeline instead of None
                    else:
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
