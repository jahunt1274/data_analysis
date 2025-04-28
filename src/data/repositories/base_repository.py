"""
Base repository for the data analysis system.

This module provides the BaseRepository abstract class that serves as the foundation
for all entity-specific repositories. It defines common read operations and utility
methods for working with static data sets.
"""

from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Optional,
    TypeVar,
    Generic,
    Type,
    Any,
    Union,
    Tuple,
    Iterator
)
import json
import logging
from datetime import datetime
from pathlib import Path
import functools

from pydantic import BaseModel

from src.data.models.base_model import ObjectId

# Type variable for the model type
T = TypeVar("T", bound=BaseModel)


class BaseRepository(Generic[T], ABC):
    """
    Base repository for data access.

    This abstract class provides common read operations and utility methods
    for working with static data through Pydantic models.

    Attributes:
        _collection_name (str): Name of the data collection
        _model_class (Type[T]): Pydantic model class for this repository
        _db (Optional[Any]): Database connection object (to be set by subclasses)
        _cache (Dict): In-memory cache for frequently accessed data
    """

    def __init__(self, collection_name: str, model_class: Type[T]):
        """
        Initialize the repository.

        Args:
            collection_name: Name of the data collection
            model_class: Pydantic model class to use for this repository
        """
        self._collection_name = collection_name
        self._model_class = model_class
        self._db = None
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._cache = {}
        self._data_loaded = False
        self._all_data = None  # Cache for all documents in the collection

    @abstractmethod
    def connect(self) -> None:
        """
        Connect to the database.

        This method must be implemented by subclasses to establish
        the database connection.
        """
        pass

    @property
    def collection(self) -> Any:
        """
        Get the data collection.

        Returns:
            The data collection object

        Raises:
            RuntimeError: If the database connection is not established
        """
        if self._db is None:
            raise RuntimeError(
                "Database connection not established. Call connect() first."
            )
        return self._db[self._collection_name]

    def _convert_id(self, id_value: Union[str, Dict, ObjectId]) -> Dict:
        """
        Convert an ID value to the appropriate format.

        Args:
            id_value: ID value to convert (can be string, dict with $oid, or ObjectId)

        Returns:
            Dict: ID in the appropriate format
        """
        if isinstance(id_value, str):
            return {"$oid": id_value}
        elif isinstance(id_value, dict) and "$oid" in id_value:
            return id_value
        elif isinstance(id_value, ObjectId):
            return {"$oid": id_value.oid}

        # If none of the above, assume it's already in the correct format
        return id_value

    def _prepare_query(self, query: Dict) -> Dict:
        """
        Prepare a query by converting any ID fields.

        Args:
            query: Query dictionary

        Returns:
            Dict: Prepared query
        """
        prepared_query = {}

        for key, value in query.items():
            if key == "_id" or key.endswith("_id"):
                if value is not None:
                    prepared_query[key] = self._convert_id(value)
            else:
                prepared_query[key] = value

        return prepared_query

    def _to_model(self, data: Dict) -> T:
        """
        Convert raw data to a Pydantic model.

        Args:
            data: Raw data

        Returns:
            T: Pydantic model instance
        """
        try:
            # Convert data to model
            return self._model_class.model_validate(data)
        except Exception as e:
            self._logger.error(f"Error converting data to model: {e}")
            # Attempt a more lenient parsing with exclude_unset=True
            return self._model_class.model_validate(data, strict=False)

    def _to_dict(self, model: T) -> Dict:
        """
        Convert a Pydantic model to a dictionary.

        Args:
            model: Pydantic model instance

        Returns:
            Dict: Dictionary representation
        """
        # Convert model to dict with by_alias=True to preserve field names
        return model.model_dump(by_alias=True, exclude_none=True)

    # Cache decorator for query methods
    def _cache_result(func):
        """Decorator to cache results of repository methods."""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create a cache key based on function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Check if result is in cache
            if cache_key in self._cache:
                return self._cache[cache_key]

            # Call the function and cache the result
            result = func(self, *args, **kwargs)
            self._cache[cache_key] = result
            return result

        return wrapper

    @_cache_result
    def find_by_id(self, id_value: Union[str, Dict, ObjectId]) -> Optional[T]:
        """
        Find a document by its ID.

        Args:
            id_value: Document ID

        Returns:
            Optional[T]: Model instance or None if not found
        """
        query = {"_id": self._convert_id(id_value)}
        result = self.collection.find_one(query)

        if result:
            return self._to_model(result)
        return None

    @_cache_result
    def find_one(self, query: Dict) -> Optional[T]:
        """
        Find a single document matching the query.

        Args:
            query: Query dictionary

        Returns:
            Optional[T]: Model instance or None if not found
        """
        prepared_query = self._prepare_query(query)
        result = self.collection.find_one(prepared_query)

        if result:
            return self._to_model(result)
        return None

    @_cache_result
    def find_many(
        self,
        query: Dict,
        skip: int = 0,
        limit: Optional[int] = None,
        sort_by: Optional[Union[str, List[Tuple[str, int]]]] = None,
        sort_direction: int = 1,  # 1 for ascending, -1 for descending
    ) -> List[T]:
        """
        Find multiple documents matching the query.

        Args:
            query: Query dictionary
            skip: Number of documents to skip
            limit: Maximum number of documents to return (None for all)
            sort_by: Field to sort by or list of (field, direction) tuples
            sort_direction: Sort direction if sort_by is a string (1=ascending, -1=descending)

        Returns:
            List[T]: List of model instances
        """
        prepared_query = self._prepare_query(query)
        cursor = self.collection.find(prepared_query).skip(skip)

        if limit is not None:
            cursor = cursor.limit(limit)

        if sort_by:
            if isinstance(sort_by, str):
                cursor = cursor.sort(sort_by, sort_direction)
            else:
                # Sort by multiple fields
                cursor = cursor.sort(sort_by)

        return [self._to_model(doc) for doc in cursor]

    def count(self, query: Dict) -> int:
        """
        Count documents matching the query.

        Args:
            query: Query dictionary

        Returns:
            int: Number of matching documents
        """
        prepared_query = self._prepare_query(query)
        return self.collection.count_documents(prepared_query)

    def aggregate(self, pipeline: List[Dict]) -> List[Dict]:
        """
        Perform an aggregation operation.

        Args:
            pipeline: Aggregation pipeline

        Returns:
            List[Dict]: Results of the aggregation
        """
        try:
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            self._logger.error(f"Error performing aggregation: {e}")
            return []

    def load_data_from_file(self, filepath: str) -> int:
        """
        Load data from a JSON file into the collection.
        Used for initial data loading only.

        Args:
            filepath: Path to the JSON file

        Returns:
            int: Number of documents loaded
        """
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Create a new empty collection (drop if exists)
            if self._collection_name in self._db.list_collection_names():
                self._db.drop_collection(self._collection_name)

            collection = self._db[self._collection_name]

            # Insert the data
            if isinstance(data, list):
                if data:
                    collection.insert_many(data)
                    return len(data)
            else:
                # Single document
                collection.insert_one(data)
                return 1

            return 0
        except Exception as e:
            self._logger.error(f"Error loading data from file {filepath}: {e}")
            return 0

    def load_data_from_directory(
        self, directory_path: str, file_pattern: str = "*.json"
    ) -> Dict[str, int]:
        """
        Load data from all JSON files in a directory.

        Args:
            directory_path: Path to the directory
            file_pattern: File pattern to match (default: "*.json")

        Returns:
            Dict[str, int]: Dictionary mapping filenames to number of documents loaded
        """
        results = {}
        try:
            # Get all JSON files in the directory
            path = Path(directory_path)
            json_files = list(path.glob(file_pattern))

            for json_file in json_files:
                count = self.load_data_from_file(str(json_file))
                results[json_file.name] = count

            return results
        except Exception as e:
            self._logger.error(
                f"Error loading data from directory {directory_path}: {e}"
            )
            return results

    @_cache_result
    def get_all(self, skip: int = 0, limit: Optional[int] = None) -> List[T]:
        """
        Get all documents from the collection.

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return (None for all)

        Returns:
            List[T]: List of model instances
        """
        return self.find_many({}, skip=skip, limit=limit)

    def get_all_lazy(self) -> Iterator[T]:
        """
        Get all documents as a lazy iterator to handle large collections.

        Returns:
            Iterator[T]: Iterator of model instances
        """
        cursor = self.collection.find({})
        for doc in cursor:
            yield self._to_model(doc)

    @_cache_result
    def get_distinct_values(
        self, field: str, query: Optional[Dict] = None
    ) -> List[Any]:
        """
        Get distinct values for a field.

        Args:
            field: Field to get distinct values for
            query: Optional query to filter documents

        Returns:
            List[Any]: List of distinct values
        """
        prepared_query = self._prepare_query(query) if query else {}
        return self.collection.distinct(field, prepared_query)

    @_cache_result
    def get_date_range(
        self, date_field: str
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the range of dates for a date field.

        Args:
            date_field: Name of the date field

        Returns:
            Tuple[Optional[datetime], Optional[datetime]]: Min and max dates
        """
        try:
            # Get the minimum date
            min_result = self.aggregate(
                [
                    {"$sort": {date_field: 1}},
                    {"$limit": 1},
                    {"$project": {date_field: 1}},
                ]
            )

            # Get the maximum date
            max_result = self.aggregate(
                [
                    {"$sort": {date_field: -1}},
                    {"$limit": 1},
                    {"$project": {date_field: 1}},
                ]
            )

            min_date = min_result[0][date_field] if min_result else None
            max_date = max_result[0][date_field] if max_result else None

            # Convert to datetime objects if they are strings
            if isinstance(min_date, str):
                min_date = datetime.fromisoformat(min_date.replace("Z", "+00:00"))
            if isinstance(max_date, str):
                max_date = datetime.fromisoformat(max_date.replace("Z", "+00:00"))

            return min_date, max_date
        except Exception as e:
            self._logger.error(f"Error getting date range for field {date_field}: {e}")
            return None, None

    def get_field_statistics(
        self, field: str, query: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for a numeric field (min, max, avg, etc.).

        Args:
            field: Field to get statistics for
            query: Optional query to filter documents

        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        try:
            prepared_query = self._prepare_query(query) if query else {}

            pipeline = [
                {"$match": prepared_query},
                {
                    "$group": {
                        "_id": None,
                        "min": {"$min": f"${field}"},
                        "max": {"$max": f"${field}"},
                        "avg": {"$avg": f"${field}"},
                        "sum": {"$sum": f"${field}"},
                        "count": {"$sum": 1},
                    }
                },
            ]

            result = self.aggregate(pipeline)
            if result:
                return result[0]
            return {"count": 0}
        except Exception as e:
            self._logger.error(f"Error getting statistics for field {field}: {e}")
            return {"count": 0}

    @_cache_result
    def get_related_documents(
        self, field: str, value: Any, target_repo: "BaseRepository"
    ) -> List[Any]:
        """
        Get documents from another repository that relate to documents in this repository.

        Args:
            field: Field in the target repository to match
            value: Value to match in the target repository
            target_repo: Target repository to query

        Returns:
            List[Any]: List of related documents from the target repository
        """
        return target_repo.find_many({field: value})

    def clear_cache(self) -> None:
        """
        Clear the cache.
        """
        self._cache.clear()

    def group_by(
        self,
        group_field: str,
        query: Optional[Dict] = None,
        count_field: Optional[str] = None,
    ) -> Dict[Any, Union[int, List[T]]]:
        """
        Group documents by a field.

        Args:
            group_field: Field to group by
            query: Optional query to filter documents
            count_field: If specified, count by this field instead of documents

        Returns:
            Dict[Any, Union[int, List[T]]]: Dictionary mapping group values to counts or document lists
        """
        try:
            prepared_query = self._prepare_query(query) if query else {}

            if count_field:
                # Group and count by specified field
                pipeline = [
                    {"$match": prepared_query},
                    {
                        "$group": {
                            "_id": f"${group_field}",
                            "count": {"$sum": f"${count_field}"},
                        }
                    },
                    {"$sort": {"_id": 1}},
                ]

                result = self.aggregate(pipeline)
                return {doc["_id"]: doc["count"] for doc in result}
            else:
                # Group and return documents
                docs = self.find_many(prepared_query)
                grouped = {}

                for doc in docs:
                    key = getattr(doc, group_field, None)
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(doc)

                return grouped
        except Exception as e:
            self._logger.error(f"Error grouping by field {group_field}: {e}")
            return {}

    def filter_by_date_range(
        self,
        date_field: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[T]:
        """
        Filter documents by a date range.

        Args:
            date_field: Name of the date field
            start_date: Starting date (inclusive)
            end_date: Ending date (inclusive)

        Returns:
            List[T]: Documents within the date range
        """
        query = {}

        if start_date or end_date:
            date_query = {}

            if start_date:
                date_query["$gte"] = start_date

            if end_date:
                date_query["$lte"] = end_date

            query[date_field] = date_query

        return self.find_many(query)

    def create_index(self, field: str, unique: bool = False) -> None:
        """
        Create an index for faster queries.
        This is a one-time operation typically called during initial data loading.

        Args:
            field: Field to index
            unique: Whether values should be unique
        """
        try:
            self.collection.create_index(field, unique=unique)
        except Exception as e:
            self._logger.error(f"Error creating index on field {field}: {e}")

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the data in the collection.

        Returns:
            Dict[str, Any]: Summary information
        """
        try:
            total_count = self.count({})

            return {
                "collection": self._collection_name,
                "document_count": total_count,
                "model_type": self._model_class.__name__,
            }
        except Exception as e:
            self._logger.error(f"Error getting data summary: {e}")
            return {"error": str(e)}

    def create_lookup_index(self, field_map: Dict[str, Any]) -> Dict[Any, List[T]]:
        """
        Create an in-memory lookup index for faster access by field values.
        Useful for creating relationships between data sets.

        Args:
            field_map: Map of field name to values

        Returns:
            Dict[Any, List[T]]: Dictionary mapping field values to documents
        """
        result = {}
        all_docs = self.get_all()

        for doc in all_docs:
            for field, _ in field_map.items():
                value = getattr(doc, field, None)
                if value is not None:
                    if value not in result:
                        result[value] = []
                    result[value].append(doc)

        return result
