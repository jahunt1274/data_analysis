"""
In-memory database implementation for the data analysis system.

This module provides a MongoDB-like interface for in-memory data storage and querying,
allowing repositories to work with static JSON data as if it were in a database.
"""

import logging
from typing import Dict, List, Any, Optional, Union


class InMemoryCollection:
    """
    In-memory collection for the in-memory database.

    Provides MongoDB-like query methods for in-memory data.
    """

    def __init__(self, name):
        """
        Initialize the collection.

        Args:
            name: Collection name
        """
        self.name = name
        self.documents = []
        self._logger = logging.getLogger(f"{self.__class__.__name__}.{name}")

    def insert_one(self, document):
        """
        Insert a single document.

        Args:
            document: Document to insert
        """
        self.documents.append(document)

    def insert_many(self, documents):
        """
        Insert multiple documents.

        Args:
            documents: List of documents to insert
        """
        self.documents.extend(documents)

    def find_one(self, query):
        """
        Find a single document matching the query.

        Args:
            query: Query to filter documents

        Returns:
            dict: Matching document or None
        """
        for doc in self.documents:
            if self._matches(doc, query):
                return doc
        return None

    def find(self, query=None):
        """
        Find all documents matching the query.

        Args:
            query: Query to filter documents

        Returns:
            InMemoryCursor: Cursor for the matching documents
        """
        if query is None:
            query = {}

        matches = [doc for doc in self.documents if self._matches(doc, query)]
        return InMemoryCursor(matches)

    def count_documents(self, query):
        """
        Count documents matching the query.

        Args:
            query: Query to filter documents

        Returns:
            int: Number of matching documents
        """
        matches = [doc for doc in self.documents if self._matches(doc, query)]
        return len(matches)

    def distinct(self, field, query=None):
        """
        Get distinct values for a field.

        Args:
            field: Field name
            query: Optional query to filter documents

        Returns:
            List: List of distinct values
        """
        if query is None:
            query = {}

        matches = [doc for doc in self.documents if self._matches(doc, query)]

        # Extract field values
        values = []
        for doc in matches:
            value = self._get_field_value(doc, field)
            if value is not None and value not in values:
                values.append(value)

        return values

    def create_index(self, field, unique=False):
        """
        Create an index (no-op in this implementation).

        Args:
            field: Field to index
            unique: Whether values should be unique
        """
        # No-op in this simple implementation
        pass

    def aggregate(self, pipeline):
        """
        Perform an aggregation operation.

        This is a simplified implementation that supports only
        basic $match, $group, $sort, and $project operations.

        Args:
            pipeline: Aggregation pipeline

        Returns:
            List: Results of the aggregation
        """
        results = self.documents

        for stage in pipeline:
            if "$match" in stage:
                results = [
                    doc for doc in results if self._matches(doc, stage["$match"])
                ]
            elif "$group" in stage:
                results = self._group(results, stage["$group"])
            elif "$sort" in stage:
                results = self._sort(results, stage["$sort"])
            elif "$project" in stage:
                results = self._project(results, stage["$project"])

        return results

    def _get_field_value(self, doc, field):
        """
        Get a field value from a document.

        Supports dot notation for nested fields.

        Args:
            doc: Document
            field: Field name

        Returns:
            Field value or None
        """
        if "." in field:
            parts = field.split(".")
            value = doc
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        elif field in doc:
            return doc[field]
        return None

    def _matches(self, doc, query):
        """
        Check if a document matches a query.

        Args:
            doc: Document
            query: Query

        Returns:
            bool: True if the document matches
        """
        for key, value in query.items():
            if key.startswith("$"):
                # Logical operators
                if key == "$and":
                    if not all(self._matches(doc, q) for q in value):
                        return False
                elif key == "$or":
                    if not any(self._matches(doc, q) for q in value):
                        return False
            else:
                # Field match
                if isinstance(value, dict) and any(
                    k.startswith("$") for k in value.keys()
                ):
                    # Operator match
                    field_value = self._get_field_value(doc, key)
                    if not self._matches_operator(field_value, value):
                        return False
                else:
                    # Exact match or array membership
                    field_value = self._get_field_value(doc, key)

                    # Check for array membership
                    if isinstance(field_value, list):
                        if value not in field_value:
                            return False
                    # Regular exact match
                    elif field_value != value:
                        return False

        return True

    def _matches_operator(self, field_value, operators):
        """
        Check if a field value matches operator conditions.

        Args:
            field_value: Field value
            operators: Operator conditions

        Returns:
            bool: True if the field value matches
        """
        for op, value in operators.items():
            if op == "$eq" and field_value != value:
                return False
            elif op == "$ne" and field_value == value:
                return False
            elif op == "$gt" and (field_value is None or field_value <= value):
                return False
            elif op == "$gte" and (field_value is None or field_value < value):
                return False
            elif op == "$lt" and (field_value is None or field_value >= value):
                return False
            elif op == "$lte" and (field_value is None or field_value > value):
                return False
            elif op == "$in" and field_value not in value:
                return False
            elif op == "$nin" and field_value in value:
                return False

        return True

    def _group(self, documents, group_spec):
        """
        Perform a group operation.

        Args:
            documents: Documents to group
            group_spec: Group specification

        Returns:
            List: Grouped results
        """
        # Simple implementation for basic grouping
        groups = {}
        id_spec = group_spec.pop("_id")

        for doc in documents:
            # Determine group key
            if id_spec is None:
                key = None
            elif isinstance(id_spec, str) and id_spec.startswith("$"):
                # Group by field
                field = id_spec[1:]
                key = self._get_field_value(doc, field)
            else:
                # Complex grouping (not fully implemented)
                key = str(id_spec)

            # Create group if not exists
            if key not in groups:
                groups[key] = {"_id": key}
                for field, spec in group_spec.items():
                    if isinstance(spec, dict):
                        if "$sum" in spec:
                            groups[key][field] = 0
                        elif "$avg" in spec:
                            groups[key][field] = {"sum": 0, "count": 0}
                        elif "$min" in spec:
                            groups[key][field] = None
                        elif "$max" in spec:
                            groups[key][field] = None

            # Accumulate values
            for field, spec in group_spec.items():
                if isinstance(spec, dict):
                    if "$sum" in spec:
                        if spec["$sum"] == 1:
                            groups[key][field] += 1
                        else:
                            value = self._get_field_value(doc, spec["$sum"][1:])
                            if value is not None:
                                groups[key][field] += value
                    elif "$avg" in spec:
                        value = self._get_field_value(doc, spec["$avg"][1:])
                        if value is not None:
                            groups[key][field]["sum"] += value
                            groups[key][field]["count"] += 1
                    elif "$min" in spec:
                        value = self._get_field_value(doc, spec["$min"][1:])
                        if value is not None and (
                            groups[key][field] is None or value < groups[key][field]
                        ):
                            groups[key][field] = value
                    elif "$max" in spec:
                        value = self._get_field_value(doc, spec["$max"][1:])
                        if value is not None and (
                            groups[key][field] is None or value > groups[key][field]
                        ):
                            groups[key][field] = value

        # Finalize averages
        for group in groups.values():
            for field, value in group.items():
                if isinstance(value, dict) and "sum" in value and "count" in value:
                    if value["count"] > 0:
                        group[field] = value["sum"] / value["count"]
                    else:
                        group[field] = 0

        return list(groups.values())

    def _sort(self, documents, sort_spec):
        """
        Sort documents.

        Args:
            documents: Documents to sort
            sort_spec: Sort specification

        Returns:
            List: Sorted documents
        """
        if isinstance(sort_spec, str):
            # Single field sorting
            field = sort_spec
            return sorted(
                documents,
                key=lambda doc: self._get_comparable_value(
                    self._get_field_value(doc, field)
                ),
            )
        elif isinstance(sort_spec, list):
            # List of (field, direction) tuples
            def sort_key(doc):
                result = []
                for field, direction in sort_spec:
                    value = self._get_field_value(doc, field)
                    comparable = self._get_comparable_value(value)
                    # Apply direction (-1 reverses the sort)
                    if direction == -1 and comparable is not None:
                        # Invert comparable values for reverse sorting
                        if isinstance(comparable, (int, float)):
                            comparable = -comparable
                        elif isinstance(comparable, str):
                            # For strings, we can use a tuple with a negative flag
                            comparable = (1, comparable)
                    result.append(comparable)
                return tuple(result)

            return sorted(documents, key=sort_key)
        else:
            # Dictionary of field: direction pairs
            def sort_key(doc):
                result = []
                for field, direction in sort_spec.items():
                    value = self._get_field_value(doc, field)
                    comparable = self._get_comparable_value(value)
                    # Apply direction (-1 reverses the sort)
                    if direction == -1 and comparable is not None:
                        # Invert comparable values for reverse sorting
                        if isinstance(comparable, (int, float)):
                            comparable = -comparable
                        elif isinstance(comparable, str):
                            # For strings, we can use a tuple with a negative flag
                            comparable = (1, comparable)
                    result.append(comparable)
                return tuple(result)

            return sorted(documents, key=sort_key)

    def _get_comparable_value(self, value):
        """
        Convert a value to a comparable form.

        This handles dictionary values and other non-comparable types.

        Args:
            value: The value to make comparable

        Returns:
            A comparable value or None
        """
        if value is None:
            return None
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, dict):
            # Convert dict to a stable string representation for comparison
            # Sort keys to ensure consistent comparison
            return str(sorted(value.items()))
        elif isinstance(value, list):
            # Convert list to tuple for comparison
            return tuple(self._get_comparable_value(item) for item in value)
        else:
            # Other types: use string representation
            return str(value)

    def _project(self, documents, project_spec):
        """
        Project documents.

        Args:
            documents: Documents to project
            project_spec: Projection specification

        Returns:
            List: Projected documents
        """
        result = []
        for doc in documents:
            projected = {}
            for field, include in project_spec.items():
                if include:
                    projected[field] = self._get_field_value(doc, field)
            result.append(projected)
        return result


class InMemoryCursor:
    """
    Cursor for in-memory query results.

    Provides MongoDB-like cursor methods.
    """

    def __init__(self, documents):
        """
        Initialize the cursor.

        Args:
            documents: Query result documents
        """
        self.documents = documents
        self.position = 0

    def __iter__(self):
        """
        Iterator protocol implementation.

        Returns:
            self: Iterator
        """
        return self

    def __next__(self):
        """
        Get the next document.

        Returns:
            dict: Next document

        Raises:
            StopIteration: When no more documents
        """
        if self.position >= len(self.documents):
            raise StopIteration

        document = self.documents[self.position]
        self.position += 1
        return document

    def skip(self, count):
        """
        Skip a number of documents.

        Args:
            count: Number of documents to skip

        Returns:
            self: Cursor
        """
        self.documents = self.documents[count:]
        return self

    def limit(self, count):
        """
        Limit the number of documents.

        Args:
            count: Maximum number of documents

        Returns:
            self: Cursor
        """
        self.documents = self.documents[:count]
        return self

    def sort(self, spec, direction=None):
        """
        Sort the documents.

        Args:
            spec: Sort specification or field name
            direction: Sort direction (1=ascending, -1=descending)

        Returns:
            self: Cursor
        """
        if isinstance(spec, str):
            reverse = direction == -1
            self.documents.sort(key=lambda doc: doc.get(spec), reverse=reverse)
        else:
            # Multiple sort fields
            def sort_key(doc):
                if isinstance(spec, list):
                    return tuple(doc.get(field) for field, _ in spec)
                return tuple(doc.get(field) for field in spec.keys())

            reverse = False  # Not fully implemented for multi-field sort
            self.documents.sort(key=sort_key, reverse=reverse)

        return self


class InMemoryDatabase:
    """
    Simple in-memory database implementation.

    Provides a MongoDB-like interface for storing and querying JSON data.
    """

    def __init__(self):
        """Initialize the in-memory database."""
        self.collections = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def __getitem__(self, collection_name):
        """
        Get a collection by name.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection object
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = InMemoryCollection(collection_name)
        return self.collections[collection_name]

    def list_collection_names(self):
        """
        Get a list of all collection names.

        Returns:
            List[str]: List of collection names
        """
        return list(self.collections.keys())

    def drop_collection(self, collection_name):
        """
        Drop a collection.

        Args:
            collection_name: Name of the collection to drop
        """
        if collection_name in self.collections:
            del self.collections[collection_name]

    def insert_one(self, collection_name, document):
        """
        Insert a single document into a collection.

        Args:
            collection_name: Collection name
            document: Document to insert
        """
        self[collection_name].insert_one(document)

    def insert_many(self, collection_name, documents):
        """
        Insert multiple documents into a collection.

        Args:
            collection_name: Collection name
            documents: List of documents to insert
        """
        self[collection_name].insert_many(documents)

    def count(self, collection_name, query):
        """
        Count documents in a collection.

        Args:
            collection_name: Collection name
            query: Query to filter documents

        Returns:
            int: Number of matching documents
        """
        return self[collection_name].count_documents(query)
