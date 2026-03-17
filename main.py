from pyopf.io import load
from pyopf.resolve import resolve
from pyopf.uid64 import Uid64
import rerun as rr
import numpy as np
from PIL import Image
from pathlib import Path

project_path = Path("opf2/project.opf")

project = load(str(project_path))
project = resolve(project)


rr.init("OPF Viewer", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


points = project.point_cloud_objs[0].nodes[0]

rr.log("world/points", rr.Points3D(points.position, colors=points.color))

sensor_map = {sensor.id: sensor for sensor in project.input_cameras.sensors}
calib_sensor_map = {
    sensor.id: sensor for sensor in project.calibration.calibrated_cameras.sensors
}

positions = []

for i, (camera, calib_camera) in enumerate(
    zip(
        project.camera_list.cameras,
        project.calibration.calibrated_cameras.cameras,
        strict=False,
    )
):
    if not str(camera.uri).endswith(".JPG"):
        print("SKIPPED")
        continue

    positions.append(calib_camera.position)
    rr.set_time("image", sequence=i)
    entity = "world/camera"
    sensor = sensor_map[calib_camera.sensor_id]
    calib_sensor = calib_sensor_map[calib_camera.sensor_id]

    omega, phi, kappa = np.deg2rad(calib_camera.orientation_deg)

    rot = (
        np.array(
            [
                [1, 0, 0],
                [0, np.cos(omega), -np.sin(omega)],
                [0, np.sin(omega), np.cos(omega)],
            ]
        )
        @ np.array(
            [
                [np.cos(phi), 0, np.sin(phi)],
                [0, 1, 0],
                [-np.sin(phi), 0, np.cos(phi)],
            ]
        )
        @ np.array(
            [
                [np.cos(kappa), -np.sin(kappa), 0],
                [np.sin(kappa), np.cos(kappa), 0],
                [0, 0, 1],
            ]
        )
    )

    rr.log(entity, rr.Transform3D(translation=calib_camera.position, mat3x3=rot))

    rr.log(
        entity + "/image",
        rr.Pinhole(
            resolution=sensor.image_size_px,
            focal_length=calib_sensor.internals.focal_length_px,
            principal_point=calib_sensor.internals.principal_point_px,
            camera_xyz=rr.ViewCoordinates.RUB,
        ),
    )

    img_path = project_path.parent / "images" / Path(camera.uri).name
    print(img_path)
    with Image.open(img_path) as img:
        rr.log(entity + "/image/rgb", rr.Image(img))

rr.log("world/camera_path", rr.LineStrips3D([np.array(positions)]), static=True)
