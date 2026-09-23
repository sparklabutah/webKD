import re
from typing import List, Dict, Any, Union, Tuple, Optional, Callable
from types import SimpleNamespace


class XMLParser:
    def __init__(self, fields: List[Union[str, Tuple[str, ...]]]):
        """Initialize the parser with field definitions."""
        self._fields: List[Tuple[str, List[str]]] = []
        seen = set()
        for field in fields:
            if isinstance(field, str):
                canonical = field
                alternatives = [field]
            elif isinstance(field, tuple):
                if not field:
                    raise ValueError("Field tuple cannot be empty.")
                canonical = field[0]
                if not all(isinstance(alt, str) for alt in field):
                    raise TypeError("All alternatives in a tuple must be strings.")
                alternatives = list(field)
            else:
                raise TypeError("Each field must be a string or a tuple of strings.")
            if canonical in seen:
                raise ValueError(f"Duplicate field name: {canonical}")
            seen.add(canonical)
            self._fields.append((canonical, alternatives))

    def get_xml_reward_func(self) -> Callable:
        """Return a reward function that checks for proper XML tag usage."""

        def xml_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks for proper XML tag usage in completions."""

            def count_xml(trajectory) -> float:

                model_messages = [msg for msg in trajectory if msg["role"] == "assistant"]
                if not model_messages:
                    return 0.0

                xml_scores = []
                for msg in model_messages:
                    content = msg["content"]
                    score = 0
                    total_checks = 0

                    for canonical, alternatives in self._fields:
                        for alt in alternatives:
                            if content.count(f"<{alt}>") > 0 or content.count(f"</{alt}>") > 0:
                                score += 1 - abs(content.count(f"<{alt}>") - 1)
                                score += 1 - abs(content.count(f"</{alt}>") - 1)
                                total_checks += 2

                    if total_checks > 0:
                        xml_scores.append(score / total_checks)
                    else:
                        xml_scores.append(0.0)

                if not xml_scores:
                    return 0.0
                return 0.2 * (sum(xml_scores) / len(xml_scores))

            return [count_xml(c) for c in completions]

        return xml_reward_func

    def get_format_reward_func(self) -> Callable:
        """Return a reward function that checks if messages follow the expected format."""

        def format_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks if each step follows the expected format."""

            def check_format(trajectory):

                model_messages = [msg for msg in trajectory if msg["role"] == "assistant"]
                if not model_messages:
                    return 0.0

                format_scores = []
                for msg in model_messages:
                    content = msg["content"]
                    parsed = self.parse(content)
                    parsed_no_strip = self.parse(content, strip=False)

                    has_any_field = False
                    fields_with_content = 0
                    total_fields = 0

                    expected_field_count = len(self._fields)
                    present_field_sets = set()

                    has_correct_spacing = True

                    for i, (canonical, alternatives) in enumerate(self._fields):
                        field_set_present = False
                        for alt in alternatives:
                            if hasattr(parsed, alt) and getattr(parsed, alt) is not None:
                                has_any_field = True
                                fields_with_content += 1
                                total_fields += 1
                                field_set_present = True

                                if not (
                                    hasattr(parsed_no_strip, alt)
                                    and getattr(parsed_no_strip, alt) is not None
                                ):
                                    has_correct_spacing = False
                            elif content.count(f"<{alt}>") > 0 or content.count(f"</{alt}>") > 0:
                                total_fields += 1
                                field_set_present = True

                        if field_set_present:
                            present_field_sets.add(i)

                    format_score = 0.0

                    starts_with_any_field = False
                    first_field_set = self._fields[0][1]
                    for alt in first_field_set:
                        if content.strip().startswith(f"<{alt}>"):
                            starts_with_any_field = True
                            break

                    ends_with_any_field = False
                    last_field_set = self._fields[-1][1]
                    for alt in last_field_set:
                        if content.strip().endswith(f"</{alt}>"):
                            ends_with_any_field = True
                            break

                    if has_any_field:
                        field_set_ratio = len(present_field_sets) / expected_field_count
                        format_score += 0.4 * field_set_ratio

                    if has_correct_spacing:
                        format_score += 0.2

                    if starts_with_any_field:
                        format_score += 0.2

                    if ends_with_any_field:
                        format_score += 0.2

                    format_scores.append(format_score)

                if not format_scores:
                    return 0.0
                return 0.1 * (sum(format_scores) / len(format_scores))

            return [check_format(c) for c in completions]

        return format_reward_func

    def get_fields(self) -> List[str]:
        """Return a list of the canonical field names (in order)."""
        return [canonical for canonical, _ in self._fields]

    def format(self, **kwargs) -> str:
        """Format the provided keyword arguments into an XML string."""
        parts = []
        for canonical, alternatives in self._fields:
            value = None

            if canonical in kwargs:
                value = kwargs[canonical]
            else:
                for alt in alternatives:
                    if alt in kwargs:
                        value = kwargs[alt]
                        break
            if value is None:
                raise ValueError(f"Missing value for field '{canonical}' (allowed: {alternatives})")

            parts.append(f"<{canonical}>\n{value}\n</{canonical}>")
        return "\n".join(parts)

    def parse(self, text: str, strip: bool = True) -> Any:
        """Parse the given XML string and return an object with attributes corresponding"""
        results: Dict[str, Optional[str]] = {}
        for canonical, alternatives in self._fields:
            for alt in alternatives:
                pattern = rf"<{alt}>\s*(.*?)\s*</{alt}>"
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    results[alt] = match.group(1).strip() if strip else match.group(1)
                else:
                    results[alt] = None
        return SimpleNamespace(**results)
