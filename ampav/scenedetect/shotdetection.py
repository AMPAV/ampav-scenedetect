#!/bin/env python3.12

from pathlib import Path
from typing import Any
from ampav.core.logging import LOG_FORMAT
from ampav.core.schema import ToolOutput, VideoSegments, VideoSegment, VideoSegmentType, AVMetadata
from ampav.core.media import get_frames_from_video
from time import time
import logging
import argparse
from ampav.core.schema.video import KeyFrame
from ampav.core.utils import dump_data

from scenedetect import detect, ContentDetector, AdaptiveDetector, HistogramDetector, HashDetector, ThresholdDetector
from . import __version__


def detect_shot(videofile: Path,
                 min_shot_length: float=0.5,
                 detector: str='adaptive',
                 detector_args: dict[str, Any]={}) -> ToolOutput:
    """Detect shots in a video

    Args:
        videofile (Path): Video to process
        min_shot_length (float, optional): Minimum duration of a shot. Defaults to 0.5.
        detector (str, optional): Detection algorithm to use. Can be one of: 'adaptive', 'content', 'histogram', 'hash', 'threshold'.  Defaults to 'adaptive'.
        detector_args (dict[str, Any], optional): Detector-specific args. Defaults to {}.

    Returns:
        ToolOutput: A ToolOutput with VideoSegments representing shots
    """
    # create our output structure
    output = ToolOutput(tool_name="scenedetect-shotdetect",  
                        tool_version=__version__,   
                        start_time=time(),                   
                        parameters={"min_shot_length": min_shot_length,
                                    "detector": detector,
                                    "detector_args": detector_args,
                                    "content_source": str(videofile)})

    detectors = {
        'adaptive': AdaptiveDetector,
        'content': ContentDetector,
        'histogram': HistogramDetector,
        'hash': HashDetector,
        'threshold': ThresholdDetector
    }

    # set the logging to log into our output structure
    output.setup_logging()

    avmeta = AVMetadata.from_file(videofile)
    vsegs = VideoSegments(media_duration=avmeta.duration)
    for scene in detect(videofile, detectors[detector](min_scene_len=float(min_shot_length), **detector_args), backend='pyav'):
        vsegs.segments.append(VideoSegment(start_time=scene[0].seconds,
                                           end_time=scene[1].seconds,
                                           type=VideoSegmentType.SHOT))
    logging.info(f"Finished detecting {len(vsegs.segments)} shots.")


    # now we need to go through and get all of our frame images.  We'll pick
    # times that are the midpoint of each segment
    logging.info(f"Retrieving shot key frames")
    frame_times = [(x.start_time + x.end_time)/2 for x in vsegs.segments]
    for k, v in get_frames_from_video(videofile, 0, frame_times).items():
        if v is not None:
            vsegs.segments[frame_times.index(k)].keyframes.append(KeyFrame(time=k, frame=v))
                                                    
    
    output.output = vsegs
    output.end_time = time()
    return output


def cli_shotdetect():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument("file", type=Path, help="File to scan for scenes")
    parser.add_argument("output", type=Path, help="Output file")
    parser.add_argument("--format", choices=['yaml', 'json', 'pickle'], default='yaml', help="Output format, default yaml")
    parser.add_argument("--min_shot_length", type=float, default=0.5, help="Minimum shot length to detect")
    parser.add_argument("--detector", choices=['adaptive', 'content', 'hash', 'histogram', 'threshold'], 
                        default='adaptive', help="Detector to use.  Default 'adaptive'")
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    logging.info("Starting processing")
    result = detect_shot(args.file, min_shot_length=args.min_shot_length, detector=args.detector)
    logging.info(f"Saving data to {args.output} in {args.format} format")
    dump_data(result, args.format, args.output)   


if __name__ == "__main__":
    cli_shotdetect()
    

