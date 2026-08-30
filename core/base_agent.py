"""
core/base_agent.py
==================
BaseAgent interface and centralized progress states.
Every agent inherits from BaseAgent and mutates only its designated section of the DIO.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ProgressState(str, Enum):
    """Standardized progress states for agent pipeline execution."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

    @classmethod
    def is_valid(cls, state: str) -> bool:
        return state in cls._value2member_map_


class BaseAgent:
    """
    Abstract base class for all Autonomous Data Analyst agents.
    
    Contract:
    - Each agent owns a specific section of the Dataset Intelligence Object (DIO).
    - An agent receives a DataFrame (or None for certain stages) and the current DIO.
    - An agent returns a tuple (df, dio) or updated DIO, recording its progress, metrics, and any errors.
    """
    name: str = "base"

    def __init__(self) -> None:
        if not self.name or self.name == "base" and type(self) is not BaseAgent:
            raise ValueError(f"Agent class {self.__class__.__name__} must define a unique 'name' attribute.")

    def run(self, df: Any, dio: Any) -> tuple[Any, Any]:
        """
        Execute the agent logic. Must be overridden by subclasses.
        
        Parameters:
            df: pandas DataFrame or data representation.
            dio: Dataset Intelligence Object (DIO dataclass or dict).
            
        Returns:
            tuple[Any, Any]: (updated_df, updated_dio)
        """
        raise NotImplementedError(f"Agent '{self.name}' must implement the run(df, dio) method.")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
