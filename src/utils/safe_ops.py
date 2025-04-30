"""
Utility functions for handling None values in data processing.

This module provides functions to safely handle operations on potentially
None values, especially string operations that commonly cause errors.
"""

import json
from datetime import datetime
from typing import Optional, Any, TypeVar, Callable, List, Dict

T = TypeVar("T")


def safe_lower(value: Optional[str]) -> str:
    """
    Safely convert a string to lowercase, handling None values.

    Args:
        value: A string or None

    Returns:
        The lowercase string if value is a string, otherwise an empty string
    """
    if value is None:
        return ""
    return str(value).lower()


def safe_str(value: Any) -> str:
    """
    Safely convert any value to a string, handling None values.

    Args:
        value: Any value that might be None

    Returns:
        A string representation or empty string if None
    """
    if value is None:
        return ""
    return str(value)


def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Safely get an attribute from an object, handling None values and missing attributes.

    Args:
        obj: The object to get the attribute from
        attr: The attribute name
        default: The default value to return if the attribute doesn't exist

    Returns:
        The attribute value or the default
    """
    if obj is None:
        return default
    return getattr(obj, attr, default)


def safe_dict_get(d: Optional[Dict], key: Any, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary, handling None values and missing keys.

    Args:
        d: The dictionary to get the value from
        key: The key to look up
        default: The default value to return if the key doesn't exist

    Returns:
        The value or the default
    """
    if d is None:
        return default
    return d.get(key, default)


def safe_apply(
    value: Optional[T], func: Callable[[T], Any], default: Any = None
) -> Any:
    """
    Safely apply a function to a value, handling None values.

    Args:
        value: The value to apply the function to
        func: The function to apply
        default: The default value to return if value is None

    Returns:
        The result of applying the function or the default
    """
    if value is None:
        return default
    return func(value)


def safe_compare(value1: Any, value2: Any, case_sensitive: bool = False) -> bool:
    """
    Safely compare two values that might be None.

    Args:
        value1: First value
        value2: Second value
        case_sensitive: Whether string comparison should be case-sensitive

    Returns:
        True if values are equal (after normalization if strings), False otherwise
    """
    if value1 is None and value2 is None:
        return True
    if value1 is None or value2 is None:
        return False

    # Convert to strings for comparison
    str1 = str(value1)
    str2 = str(value2)

    if not case_sensitive:
        str1 = str1.lower()
        str2 = str2.lower()

    return str1 == str2


def safe_filter_none(items: Optional[List[T]]) -> List[T]:
    """
    Safely filter None values from a list, handling None list.

    Args:
        items: A list that might contain None values or be None itself

    Returns:
        A list with None values removed, or an empty list if input is None
    """
    if items is None:
        return []
    return [item for item in items if item is not None]


def safe_enum_from_string(
    enum_class: Any, value: Optional[str], default: Any = None
) -> Any:
    """
    Safely convert a string to an enum value, handling None and invalid values.

    Args:
        enum_class: The enum class
        value: The string value to convert
        default: The default value to return if conversion fails

    Returns:
        The enum value or the default
    """
    if value is None:
        return default

    value_str = str(value).lower()

    for member in enum_class:
        if safe_lower(member.value) == value_str:
            return member

    return default


def safe_normalize_date(date_input, output_format='%Y-%m-%d %H:%M:%S'):
    """
    Normalize a date from various formats into a specified output format.
    
    Args:
        date_input: The date to normalize, can be one of:
            - integer (epoch time in milliseconds)
            - dict with $date key containing ISO format date string
            - string in ISO format
        output_format: The desired output format (default: '%Y-%m-%d %H:%M:%S')
            
    Returns:
        A string representation of the date in the specified format
        
    Raises:
        ValueError: If the date_input is not in a recognized format
    """
    # Epoch time in milliseconds (integer)
    if isinstance(date_input, (int, float)) or (isinstance(date_input, str) and date_input.isdigit()):
        try:
            # Convert to integer in case it's a string
            epoch_ms = int(date_input)
            # Convert milliseconds to seconds
            dt = datetime.fromtimestamp(epoch_ms / 1000.0)
            return dt.strftime(output_format)
        except (ValueError, OverflowError) as e:
            # If the number is too large or otherwise invalid
            raise ValueError(f"Invalid epoch timestamp: {e}")
    
    # Object with $date key
    if isinstance(date_input, dict) and '$date' in date_input:
        try:
            # Parse the ISO format date string
            dt = datetime.fromisoformat(date_input['$date'].replace('Z', '+00:00'))
            return dt.strftime(output_format)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid date object: {e}")
    
    # ISO format date string
    if isinstance(date_input, str):
        try:
            # Try to parse as ISO format
            # Replace 'Z' with '+00:00' for UTC timezone compatibility
            dt = datetime.fromisoformat(date_input.replace('Z', '+00:00'))
            return dt.strftime(output_format)
        except ValueError as e:
            # Not an ISO format string
            pass
            
        # Try handling JSON string that contains a date object
        try:
            parsed_json = json.loads(date_input)
            if isinstance(parsed_json, dict) and '$date' in parsed_json:
                return safe_normalize_date(parsed_json, output_format)
        except (json.JSONDecodeError, TypeError):
            # Not a valid JSON string
            pass
    
    # If we got here, the format wasn't recognized
    raise ValueError(f"Unrecognized date format: {date_input}")

def safe_date_comparison(date1, date2):
    """
    Safely compare two date objects, handling the case where one might be a dict.

    Args:
        date1: First date object or dict
        date2: Second date object or dict

    Returns:
        int: -1 if date1 < date2, 0 if equal, 1 if date1 > date2, None if incomparable
    """

    # If either is a dict, extract date from it or use the object itself
    def get_date_value(date_obj):
        if isinstance(date_obj, dict):
            # Try to extract date from common dict formats
            if "$date" in date_obj:
                date_str = date_obj["$date"]
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return None
        return date_obj

    date1_value = get_date_value(date1)
    date2_value = get_date_value(date2)

    # If both are valid datetime objects, compare them
    if isinstance(date1_value, datetime) and isinstance(date2_value, datetime):
        if date1_value < date2_value:
            return -1
        elif date1_value > date2_value:
            return 1
        else:
            return 0

    # If one is not a valid datetime, return None to indicate incomparable
    return None
