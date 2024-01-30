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
        else: self.extract() 

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
        extracted_path = os.path.splitext(os.path.sep.join(edf.path.split(os.path.sep)[-2:]))[0] + ".csv"
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
        raw = mne.io.read_raw_edf(read)
        # map channel names to the ones extracted in summary
        try:
            channel_mapping = dict(zip(raw.ch_names, edf.channels))
        except:
            print(f"    ERROR MAPPING CHANNELS FOR {edf.name}")
            raise RuntimeError 


        try:
            raw.rename_channels(channel_mapping)
        except:
            control.warning(f"    ERROR RENAMING CHANNELS (MNE) FOR {edf.name}, CHANNELS:\n {edf.channels}\nRAW CHANNELS:\n {raw.ch_names}")
            if None in channel_mapping.values():
                # where raw_channel_name is the ones loaded from the .edf file
                # and set_channel_name is the ones extracted from the summary
                # and currently zipped together 
                for raw_channel_name, set_channel_name in channel_mapping.items():
                    if set_channel_name == None:
                        control.warning(f"        setting missing channel name to {raw_channel_name}")
                        channel_mapping[raw_channel_name] = raw_channel_name
                try:
                    raw.rename_channels(channel_mapping) 
                except ValueError:
                    control.warning(f"        issues when extracting the channel mapping for {edf.name}, falling back to raw channel names.")


        # extract EEG datapoints to a pandas dataframe
        all_channels = pd.DataFrame()
        for channel in raw.ch_names:
            if channel == None: break
            selected_channel = raw.copy().pick(channel)
            data, times = selected_channel[:, :]
            df_channel = pd.DataFrame(data.T, columns=[channel])
            all_channels = pd.concat([all_channels, df_channel], axis=1)

        # add in classification column
        classification_df = pd.DataFrame([False] * all_channels.shape[0], columns=["Preictal"])
        for period, times in edf.preictal_period_dict.items():
            starting_hz = int(times["start"]) * edf.data_sampling_rate
            ending_hz = int(times["end"]) * edf.data_sampling_rate
            for index, value in classification_df['Preictal'].items():
                if starting_hz < index  and index < ending_hz:
                    classification_df.loc[index, "Preictal"] = True


        final_df = pd.concat([classification_df,  all_channels], axis=1)

        # write dataframe to disc
        print(f"EXTRACTOR : writing {edf.name} to {write}")
        final_df.to_csv(write, index=False)
        print(f"EXTRACTOR : Finished {edf.name}")


