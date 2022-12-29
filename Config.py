from Commons.mod import resource_path
from VirtualScreen.Calibration import Parameter3D, FixedParameter
from Commons.vector import Vector2D, Vector3D
from Commons.landmarks import Landmark, Landmarks, ScreenLandmarks, LandmarkPoint

import numpy as np

class Config:

    def __init__(self) -> None:
        self.__file = open(resource_path('./config.txt'), 'a+')
        self.data = {}
        self.__read()

    def __read(self):
        self.__file.seek(0)
        data = self.__file.read()
        data_list = data.split()
        for d in data_list:
            key_value = d.split('=')
            if len(key_value) < 2:
                continue
            self.data[key_value[0]] = key_value[1]

    def write(self):
        self.__file.truncate(0)
        text = ""
        for key in self.data:
            if self.data[key] == None:
                continue
            text = text + f"{key}={self.data[key]}\n"
        self.__file.write(text)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__file.close()

class ControllerState:
    def __init__(self) -> None:
        self.config = Config()
        self.mode = self.config.data["mode"] if "mode" in self.config.data else "relative"
        self.calibrating = False
        self.lefty = self.config.data["lefty"] == "True" if "lefty" in self.config.data else False
        self.contrast = float(self.config.data["contrast"]) if "contrast" in self.config.data else 1.0
        self.brightness = float(self.config.data["brightness"]) if "brightness" in self.config.data else 0
        self.fixedParameter = self.__getFixedParameter()
        self.screenLandmarks = self.__getScreenLandmarks()

    def __getFixedParameter(self):
        hasCentorPoint = "FixedParameter_CentorPoint_X" in self.config.data and "FixedParameter_CentorPoint_Y" in self.config.data and "FixedParameter_CentorPoint_Z" in self.config.data
        hasOriginVector = "FixedParameter_OriginVector_X" in self.config.data and "FixedParameter_OriginVector_Y" in self.config.data and "FixedParameter_OriginVector_Z" in self.config.data
        hasHorizontalDirection = "FixedParameter_HorizontalDirection_X" in self.config.data and "FixedParameter_HorizontalDirection_Y" in self.config.data and "FixedParameter_HorizontalDirection_Z" in self.config.data
        if not (hasCentorPoint and hasOriginVector and hasHorizontalDirection):
            return None
        return FixedParameter(
            centor_point=Parameter3D(
                x=np.array([float(num) for num in self.config.data["FixedParameter_CentorPoint_X"].split(',')]),
                y=np.array([float(num) for num in self.config.data["FixedParameter_CentorPoint_Y"].split(',')]),
                z=np.array([float(self.config.data["FixedParameter_CentorPoint_Z"])])
            ),
            origin_vector=Parameter3D(
                x=np.array([float(self.config.data["FixedParameter_OriginVector_X"])]),
                y=np.array([float(self.config.data["FixedParameter_OriginVector_Y"])]),
                z=np.array([float(num) for num in self.config.data["FixedParameter_OriginVector_Z"].split(',')])
            ),
            horizontal_direction=Parameter3D(
                x=float(self.config.data["FixedParameter_HorizontalDirection_X"]),
                y=float(self.config.data["FixedParameter_HorizontalDirection_Y"]),
                z=float(self.config.data["FixedParameter_HorizontalDirection_Z"])
            ),
        )

    def __getScreenLandmarks(self):
        hasEye = "ScreenLandmarks_Eye_X" in self.config.data and "ScreenLandmarks_Eye_Y" in self.config.data and "ScreenLandmarks_Eye_Z" in self.config.data
        hasOriginPoint = "ScreenLandmarks_OriginPoint_X" in self.config.data and "ScreenLandmarks_OriginPoint_Y" in self.config.data and "ScreenLandmarks_OriginPoint_Z" in self.config.data
        hasDiagonalPoint = "ScreenLandmarks_DiagonalPoint_X" in self.config.data and "ScreenLandmarks_DiagonalPoint_Y" in self.config.data and "ScreenLandmarks_DiagonalPoint_Z" in self.config.data
        hasHorizontalDirection = "ScreenLandmarks_HorizontalDirection_X" in self.config.data and "ScreenLandmarks_HorizontalDirection_Y" in self.config.data and "ScreenLandmarks_HorizontalDirection_Z" in self.config.data
        hasScale = "ScreenLandmarks_Scale_X" in self.config.data and "ScreenLandmarks_Scale_Y" in self.config.data
        if not (hasEye and hasOriginPoint and hasDiagonalPoint and hasHorizontalDirection and hasScale):
            return None
        return ScreenLandmarks(
            eye=Vector3D(x=float(self.config.data["ScreenLandmarks_Eye_X"]), y=float(self.config.data["ScreenLandmarks_Eye_Y"]), z=float(self.config.data["ScreenLandmarks_Eye_Z"])),
            origin_point=Vector3D(x=float(self.config.data["ScreenLandmarks_OriginPoint_X"]), y=float(self.config.data["ScreenLandmarks_OriginPoint_Y"]), z=float(self.config.data["ScreenLandmarks_OriginPoint_Z"])),
            diagonal_point=Vector3D(x=float(self.config.data["ScreenLandmarks_DiagonalPoint_X"]), y=float(self.config.data["ScreenLandmarks_DiagonalPoint_Y"]), z=float(self.config.data["ScreenLandmarks_DiagonalPoint_Z"])),
            horizontal_direction=Vector3D(x=float(self.config.data["ScreenLandmarks_HorizontalDirection_X"]), y=float(self.config.data["ScreenLandmarks_HorizontalDirection_Y"]), z=float(self.config.data["ScreenLandmarks_HorizontalDirection_Z"])),
            scale=Vector2D(x=float(self.config.data["ScreenLandmarks_Scale_X"]), y=float(self.config.data["ScreenLandmarks_Scale_Y"])),
        )
    
    def setFixedparameter(self, fixedParameter: FixedParameter):
        self.fixedParameter = fixedParameter
        self.config.data["FixedParameter_CentorPoint_X"] = ','.join(map(str, self.fixedParameter.centor_point.x.tolist())) if fixedParameter else None
        self.config.data["FixedParameter_CentorPoint_Y"] = ','.join(map(str, self.fixedParameter.centor_point.y.tolist())) if fixedParameter else None
        self.config.data["FixedParameter_CentorPoint_Z"] = ','.join(map(str, self.fixedParameter.centor_point.z.tolist())) if fixedParameter else None
        self.config.data["FixedParameter_OriginVector_X"] = ','.join(map(str, self.fixedParameter.origin_vector.x.tolist())) if fixedParameter else None
        self.config.data["FixedParameter_OriginVector_Y"] = ','.join(map(str, self.fixedParameter.origin_vector.y.tolist())) if fixedParameter else None
        self.config.data["FixedParameter_OriginVector_Z"] = ','.join(map(str, self.fixedParameter.origin_vector.z.tolist())) if fixedParameter else None
        self.config.data["FixedParameter_HorizontalDirection_X"] = str(self.fixedParameter.horizontal_direction.x) if fixedParameter else None
        self.config.data["FixedParameter_HorizontalDirection_Y"] = str(self.fixedParameter.horizontal_direction.y) if fixedParameter else None
        self.config.data["FixedParameter_HorizontalDirection_Z"] = str(self.fixedParameter.horizontal_direction.z) if fixedParameter else None
        self.config.write()

    def setScreenLandmarks(self, screenLandmarks: ScreenLandmarks):
        self.screenLandmarks = screenLandmarks
        self.config.data["ScreenLandmarks_Eye_X"] = str(screenLandmarks.eye.landmark.x) if screenLandmarks else None
        self.config.data["ScreenLandmarks_Eye_Y"] = str(screenLandmarks.eye.landmark.y) if screenLandmarks else None
        self.config.data["ScreenLandmarks_Eye_Z"] = str(screenLandmarks.eye.landmark.z) if screenLandmarks else None
        self.config.data["ScreenLandmarks_OriginPoint_X"] = str(screenLandmarks.origin_point.landmark.x) if screenLandmarks else None
        self.config.data["ScreenLandmarks_OriginPoint_Y"] = str(screenLandmarks.origin_point.landmark.y) if screenLandmarks else None
        self.config.data["ScreenLandmarks_OriginPoint_Z"] = str(screenLandmarks.origin_point.landmark.z) if screenLandmarks else None
        self.config.data["ScreenLandmarks_DiagonalPoint_X"] = str(screenLandmarks.diagonal_point.landmark.x) if screenLandmarks else None
        self.config.data["ScreenLandmarks_DiagonalPoint_Y"] = str(screenLandmarks.diagonal_point.landmark.y) if screenLandmarks else None
        self.config.data["ScreenLandmarks_DiagonalPoint_Z"] = str(screenLandmarks.diagonal_point.landmark.z) if screenLandmarks else None
        self.config.data["ScreenLandmarks_HorizontalDirection_X"] = str(screenLandmarks.horizontal_direction.landmark.x) if screenLandmarks else None
        self.config.data["ScreenLandmarks_HorizontalDirection_Y"] = str(screenLandmarks.horizontal_direction.landmark.y) if screenLandmarks else None
        self.config.data["ScreenLandmarks_HorizontalDirection_Z"] = str(screenLandmarks.horizontal_direction.landmark.z) if screenLandmarks else None
        self.config.data["ScreenLandmarks_Scale_X"] = str(screenLandmarks.scale.x) if screenLandmarks else None
        self.config.data["ScreenLandmarks_Scale_Y"] = str(screenLandmarks.scale.y) if screenLandmarks else None
        self.config.write()

    def setMode(self, mode):
        self.mode = mode
        self.__set("mode", mode)

    def setLefty(self, lefty):
        self.lefty = lefty
        self.__set("lefty", lefty)

    def setContrast(self, value: float):
        self.contrast = value
        self.__set("contrast", str(value))
        
    def setBrightness(self, value: float):
        self.brightness = value
        self.__set("brightness", str(value))

    def __set(self, key, value):
        self.config.data[key] = value
        self.config.write()


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.config.close()