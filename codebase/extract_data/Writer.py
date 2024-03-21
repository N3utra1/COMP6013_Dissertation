import mne
import os
import pandas as pd
import threading
import control

# from ..load_dataset.CHB import CHB 
# from ..load_dataset.EDF_File import EDF_File 

class Writer:
    # Extracts the infomration from each EDF_File contained in a CHB Object
    def __init__(self, chb=None, write_path=None, overwrite=False, threading=False):
        self.target_CHB = chb
        self.write_path = write_path
        self.overwrite = overwrite
        if threading:
            self.extract_threaded() 
        else: 
            self.extract() 

    def extract_threaded(self):
        def threaded_extraction(edf):
            self.extraction_inner(edf) 

        # Start a thread for each EDF file extraction task
        threads = []
        for edf in self.target_CHB.edf_files:
            print(f"starting thread for {edf.name}")
            thread = threading.Thread(target=threaded_extraction, args=(edf,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()
            
    def extract(self):
        for edf in self.target_CHB.edf_files:
            self.extraction_inner(edf)


    def extraction_inner(self, edf):
        def write_period(type, times, raw, frequency):
            start, end = [max(0, (time / frequency)-1) for time in times]  
            print(f"writing {type}: {times[0]}({start}s) -> {times[1]}({end})s ")
            # start, end = raw.time_as_index(times)
            if not type in ["interictal", "preictal", "ictal"]: raise ValueError 
            path = os.path.join(self.write_path, extracted_path, type, "master.csv")
            sliced_raw = raw.copy().crop(tmin=start, tmax=end)
            raw_df = sliced_raw.to_data_frame() 
            raw_df.drop("time", axis=1, inplace=True)
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path))
                raw_df.to_csv(path, index=False)
            else:
                raw_df.to_csv(path, mode="a", index=False,  header=False)
        

        extracted_path = os.path.splitext(os.path.sep.join(edf.path.split(os.path.sep)[-2:]))[0]

        read = edf.path
        write = os.path.join(self.write_path, extracted_path)

        # if the file already exists, and overwrite is false, continue
        if os.path.exists(write) and not self.overwrite:
            print(f"EXTRACTOR : overwrite set to false, skipping {edf.name}")
            return

        # print overwrite warning
        if os.path.exists(write):
            print(f"EXTRACTOR : overwrite set to true, overwriting {edf.name}")

        os.makedirs(os.path.split(write)[0], exist_ok=True)

        # read in edf file
        raw = mne.io.read_raw_edf(read, preload=True)
        # pick the common channels
        try:
            raw = raw.pick_channels(control.common_columns)
        except Exception as e:
            control.warning(f"error when writing extracted csv data to {write}. Error: {e}\n\n")
            return 

        # apply the band pass filters
        raw = raw.notch_filter(freqs=control.band_pass_low)
        raw = raw.notch_filter(freqs=control.band_pass_high)

        # seperate out ictal, preictal and interictal 
        if edf.seizure_dict == {}:
            write_period("interictal", [0, len(raw)], raw, edf.data_sampling_rate)
            return


        # compress seizures
        seizure_array = []

        for seizure_id in edf.seizure_dict:
            seizure = edf.seizure_dict[seizure_id]

            # if there have been no previous seizures, skip
            if len(seizure_array) == 0: 
                seizure_array.append(seizure)
                continue

            # if there has already been a seizure, and its within 2x the preictal period, treat it as one big seizure
            if (seizure["start"] - (((control.preictal_period * 60) * control.window_size * edf.data_sampling_rate) * 2)) < seizure_array[-1]["end"]:
                # seizure "collision"
                seizure_array[-1]["end"] = seizure["end"]
            else:
                seizure_array.append(seizure)

            print("updated seizure_array")

        current_index = 0
        for seizure in seizure_array:
            preictal_start = max(0, seizure["start"] - (control.preictal_period * 60 * edf.data_sampling_rate))
            preictal_end = seizure["start"] - 1
            write_period("preictal", [preictal_start, preictal_end], raw, edf.data_sampling_rate)

            if not preictal_start == 0:
                interictal_start = current_index + 1
                interictal_end = preictal_start - 1 
                write_period("interictal", [interictal_start, interictal_end], raw, edf.data_sampling_rate)

            ictal_start = seizure["start"]
            ictal_end = seizure["end"]
            write_period("ictal", [ictal_start, ictal_end], raw, edf.data_sampling_rate)

            current_index = seizure["end"] + 1
        # write the remainder of the file as interictal
        write_period("interictal", [current_index, len(raw)], raw, edf.data_sampling_rate)
        pass

