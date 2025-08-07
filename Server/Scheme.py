from typing import List

from pydantic import BaseModel


class CoarsePointInput(BaseModel):
    unity_point: List[float]
    point_number: int


class CenterInput(BaseModel):
    center: List[float]


class MatrixInput(BaseModel):
    matrix: List[float]
