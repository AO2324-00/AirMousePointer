from turtle import left
from typing import Optional
from landmarks import BothSides, Landmarks
from screen import SpatialPlane, calcPosition
from vector import Vector2D, Vector3D, calcDotProduct, calcVector3D, calcAngle


class Pointer:

    def __init__(self, *, config):
        self.__config = config
        self.__points = BothSides()
        self.__positions = BothSides()
        self.__saved_positions = {}
        self.__isSaving = {}

    def getPoints(self) -> BothSides:
        return self.__points
    
    def getPositions(self, raw:bool=False) -> BothSides:
        return self.__positions if raw else self.convertToRange0to1(self.__positions)

    def convertToRange0to1(self, positions):
        _positions = BothSides(left=Vector2D(x=0, y=0), right=Vector2D(x=0, y=0))
        for side in ['left', 'right']:
            position = positions.get(side)
            if not position:
                _positions.set(side, None)
                continue
            _position = _positions.get(side)
            for dir in ['x', 'y']:
                if position.get(dir) < -1:
                    _positions.set(side, None)
                    break
                elif position.get(dir) < 0:
                    _position.set(dir, 0)
                elif 2 < position.get(dir):
                    _positions.set(side, None)
                    break
                elif 1 < position.get(dir):
                    _position.set(dir, 1)
                else:
                    _position.set(dir, position.get(dir))
        return _positions

    def savePositions(self, name: str):
        self.__isSaving[name] = True
        self.__saved_positions[name] = self.__positions
    
    def releaseSavePositions(self, name: str):
        self.__isSaving[name] = False
        self.__saved_positions[name] = BothSides()

    def isSaving(self, name: str) -> bool:
        return name in self.__isSaving and self.__isSaving[name]

    def getSavePositions(self, name: str, raw:bool=False) -> BothSides:
        if name in self.__isSaving:
            return self.__saved_positions[name] if raw else self.convertToRange0to1(self.__saved_positions[name])
        else:
            return BothSides

    def calcPosition(self, spatial_plane: SpatialPlane, landmarks: Landmarks) -> BothSides:
        self.__points = BothSides()
        self.__positions = BothSides()
        if not spatial_plane:
            return BothSides()
        for side in ['left', 'right']:
            if not landmarks.hands.get(side):
                self.__points.set(side, None)
                self.__positions.set(side, None)
                continue
            ray = calcVector3D(landmarks.eye, landmarks.hands.get(side).landmark[5])
            intersection = spatial_plane.calcIntersection(landmarks.eye, ray)
            self.__points.set(side, intersection)
            position = calcPosition(spatial_plane.getVertex(), intersection)
            position.x += (1 if side == 'left' else -1) * self.__config.pointer_offset.x/self.__config.screen_size.x
            position.y -= self.__config.pointer_offset.y/self.__config.screen_size.y
            self.__positions.set(side, position)

        return self.__positions

    def getActiveSide(self, positions: BothSides) -> Optional[str]:
        side = None
        if positions.left and positions.right:
            side = 'left' if self.__config.lefty else 'right'
        elif not positions.left and not positions.right:
            side = None
        else:
            side = 'left' if positions.left else 'right'
        return side
