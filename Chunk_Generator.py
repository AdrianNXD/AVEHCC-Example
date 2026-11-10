from pathlib import Path
import soundfile as sf
import pandas as pd
import numpy as np
import os


def generate_chunks(
    t_labels:pd.DataFrame,
    location:int,
    microphone:str,
    output_folder:str,
    window_size:float           = 2.0,
    stereo_handling:str         = "right",
    init_offset:float           = 0.0,
    exclude_empty_chunks:bool   = False
):

    '''
    Splits a single recording into fixed-length audio chunks and stores them
    in the specified output folder. Multiple calls can be used to process
    additional recordings, appending their chunks to existing data.

    A file named "chunk_labels.csv" is created alongside the audio chunks.
    It contains metadata for each chunk, including the number of pass-by
    events, and can be used to generate training labels.

    Parameters
    ----------
    t_labels : pd.DataFrame
        Dataset annotations containing vehicle pass-by timestamps and
        corresponding metadata.
    location : int
        Recording location (1-15).
    microphone : str
        Microphone identifier. Available options:
        - Locations 1-8: "rd" (RODE), "se" (sE8), "gp" (GoPro).
        - Locations 9-15: "rse" (right sE8), "gp" (GoPro).
    output_folder : str
        Destination folder for the audio chunks and "chunk_labels.csv".
    window_size : float
        Chunk duration in seconds. Default: 2.0.
    stereo_handling : str
        Method for converting stereo recordings to mono:
        "left", "right" or "mean" (average of both channels).
        Default: "right".
    init_offset : float
        Initial offset in seconds before chunking starts. Can be used to
        generate an additional set of chunks shifted by half a window
        length. Default: 0.0.
    exclude_empty_chunks : bool
        If True, chunks without pass-by events are excluded. Otherwise,
        all chunks are saved. Default: False.
    '''

    samplerate = 48_000

    # Validate input parameters.
    if location not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        raise ValueError(f"Non-existing location number: {location}")

    # Check microphone availability for the selected location.
    if location <= 8 and microphone not in ["rd", "se", "gp"]:
        raise ValueError(f"There is no: {microphone} for location {location}. For location 1-8 microhpones 'rd', 'se' and 'gp' are available.")
    elif location > 8 and microphone not in ["rse", "gp"]:
        raise ValueError(f"There is no: {microphone} for location {location}. For location 9-15 microhpones 'rse' and 'gp' are available.")

    # Validate stereo handling method.
    if stereo_handling not in ["left", "right", "mean"]:
        raise ValueError(f"For stereo handling only the methods 'left', 'right', 'mean' are available and not {stereo_handling}")

    # Load existing labels or initialize a new output folder and DataFrame.
    output_folder = Path(output_folder)
    if os.path.isdir(output_folder):
        c_labels_path = output_folder / "chunk_labels.csv"
        c_labels = pd.read_csv(c_labels_path, sep=";", index_col=0)
        chunk_index = len(c_labels)
        print(f"add more chunks .. | loc: {location:3} | mic: {microphone:4} | window_size: {window_size:4} | offset: {init_offset:4}")
    else:
        print(f"create chunks .... | loc: {location:3} | mic: {microphone:4} | window_size: {window_size:4} | offset: {init_offset:4}")
        output_folder.mkdir()
        c_labels = pd.DataFrame(columns=["chunk_index", "location", "mic", "start_t", "end_t", "num_events", "direction", "category"])
        chunk_index = 0


    # Load the recording and prepare it for chunking.
    input_audio_file = f"./AVEHCC/location_{location}/{location}_{microphone}.wav"

    audio, sr = sf.read(input_audio_file)

    # Convert stereo audio to mono using the selected channel or averaging method.
    if audio.ndim == 2:
        if stereo_handling == "left":
            audio = audio[:, 0]
        elif stereo_handling == "right":
            audio = audio[:, 1]
        elif stereo_handling == "mean":
            audio = np.mean(audio, axis=1)

    # Calculate the initial chunk boundaries in seconds and samples.
    start_time = init_offset
    end_time = start_time + window_size
    start_sample = int(start_time * samplerate)
    end_sample = int(end_time * samplerate)

    # Process chunks until the remaining audio is shorter than one window.
    while(end_sample <= len(audio)):

        chunk = audio[start_sample:end_sample]

        # Retrieve all pass-by events within the current time window.
        chunk_data = t_labels[((t_labels.location == location) & (t_labels.timestamp >= start_time) & (t_labels.timestamp < end_time))]

        num_events = len(chunk_data)

        if num_events == 0 and exclude_empty_chunks:
            pass
        else:
            # Save the audio chunk.
            filename = output_folder / f"{chunk_index}.wav"
            sf.write(filename, chunk, samplerate, subtype="PCM_24")

            # Collect event directions and categories for the current chunk.
            direction = ",".join(t_labels.loc[(t_labels.location == location) & (t_labels.timestamp >= start_time) & (t_labels.timestamp < end_time), "direction"])

            category = ",".join(t_labels.loc[(t_labels.location == location) & (t_labels.timestamp >= start_time) & (t_labels.timestamp < end_time), "category"])

            # Append the chunk metadata to the label DataFrame.
            c_labels.loc[len(c_labels)] = [
                chunk_index,
                location,
                microphone,
                start_time,
                end_time,
                num_events,
                direction,
                category
            ]

            # Increment the chunk index.
            chunk_index += 1

        # Advance the time window and update the corresponding sample indices.
        start_time = start_time + window_size
        end_time = start_time + window_size
        start_sample = int(start_time * samplerate)
        end_sample = int(end_time * samplerate)

    # Save the updated chunk labels.
    c_labels.to_csv(output_folder / "chunk_labels.csv", sep=";")
