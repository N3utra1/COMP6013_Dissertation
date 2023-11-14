import mne
import os
import pandas as pd

# from ..load_dataset.CHB import CHB 
# from ..load_dataset.EDF_File import EDF_File 

class Extractor:
    # Extracts the infomration from each EDF_File contained in a CHB Object
    def __init__(self, chb=None, write_path=None, overwrite=True):
        self.target_CHB = chb
        self.write_path = write_path
        self.overwrite = overwrite
        self.extract() 

    def extract(self):
        for edf in self.target_CHB.edf_files:
            extracted_path = os.path.splitext(os.path.sep.join(edf.path.split(os.path.sep)[-2:]))[0] + ".csv"
            read = edf.path
            write = os.path.join(self.write_path, extracted_path)

            # if the file already exists, and overwrite is false, continue
            if os.path.exists(write) and not self.overwrite: 
                print(f"overwrite set to false, skipping {edf.name}")
                continue

            os.makedirs(os.path.split(write)[0], exist_ok=True)

            # read in edf file
            raw = mne.io.read_raw_edf(read)
            # map channel names to the ones extracted in summary
            try:
                channel_mapping = dict(zip(raw.ch_names, edf.channels))
            except:
                raise Exception 
            raw.rename_channels(channel_mapping)

            # extract EEG datapoints to a pandas dataframe
            all_channels = pd.DataFrame()
            for channel in edf.channels:
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
            print("STARTING WRITE")
            final_df.to_csv(write, index=False)

            break #TEMP