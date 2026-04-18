# In order to get multiple outputs on one video several steps are required:
# 1.Split the video into smaller segments using ffmpeg or any similar application.
# 2.Create a pipeline for each model
# 3.Run each model for which you want to perform identification for on separate threads using parallel processing
# 4.Combine the segments of the videos based on any metric, for example, timestamps
# 5.Merge the smaller segments of the video into a larger one using ffmpeg
# 6.Once the frames are merged, you will need to encode the video to create the final output. Utilize the video encoding capabilities of ffmpeg.

from vision.signs import detect_signs
from vision.lanes import detect_lanes
from vision.objects import detect_objects

def run_pipeline(frame):
    objects = detect_objects(frame)
    signs = detect_signs(frame)
    lanes = detect_lanes(frame)

    return {
        "objects": objects,
        "signs": signs,
        "lanes": lanes
    }

