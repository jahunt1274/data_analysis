"""
Utility functions for handling None values in data processing.

This module provides functions to safely handle operations on potentially
None values, especially string operations that commonly cause errors.
"""

from typing import Optional, Any, TypeVar, Callable, List, Dict, Union

T = TypeVar('T')


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


def safe_apply(value: Optional[T], func: Callable[[T], Any], default: Any = None) -> Any:
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


def safe_enum_from_string(enum_class: Any, value: Optional[str], default: Any = None) -> Any:
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