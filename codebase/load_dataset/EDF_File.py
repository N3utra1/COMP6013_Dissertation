import os
import re

class EDF_File():
    def __init__(self, _parent_chb, _edf_path):
        self.parent_chb = _parent_chb
        # self.full_name = _edf_path
        # self.name = _edf_path[6:]         # unknown change from linux -> windows. Could be due to malformed file names on linux machine 
        self.name = _edf_path
        self.path = os.path.join(self.parent_chb.path, _edf_path)
        self.summary_start_line = self.get_summary_start_line() 
        self.channels = self.extract_channels()
        self.data_sampling_rate = self.extract_data_sampling_rate()
        self.start_time = self.extract_start_time()
        self.end_time = self.extract_end_time() 
        self.number_of_seizures = 0 #updated in self.extract_seizures()
        self.seizure_dict = self.extract_seizures()
        self.preictal_period_dict = self.extract_preictal_period()
        
    
    def __repr__(self):
        return f"""
        Name: {self.name}
        Path: {self.path}
        Start Time: {self.start_time}
        End Time: {self.end_time}
        Channels: {self.channels}
        Data Sampling Rate (Hz): {self.data_sampling_rate}
        Number of Seizures: {self.number_of_seizures}
        Seizure Dictionary: {self.seizure_dict}
        Preictal Period Dictionary: {self.preictal_period_dict}
        """

    def get_summary_start_line(self):
        for line in self.parent_chb.full_summary:
            if re.search(f"^File Name: {self.name}.*$", line): 
                return self.parent_chb.full_summary.index(line)
            
    def extract_channels(self):
        summary = self.parent_chb.full_summary[0: self.summary_start_line] 
        i = len(summary)-1
        end_index = None
        while i  > 0:
            line = summary[i]
            if re.search("^Channel .*$", line):
                if end_index == None: end_index = i
            else:
                if not end_index == None:
                    channels = []
                    for line in summary[i + 1: end_index]:
                        name = line.split(" ")[-1]
                        count = channels.count(name)
                        if count == 0:
                            channels.append(name)
                        else:
                            channels.append(channels.append(f"{name}-{count}")) 
                    return channels
            i -= 1 
    
    def extract_data_sampling_rate(self):
        summary = self.parent_chb.full_summary[0: self.summary_start_line]
        inverted_summary = summary[::-1] 
        for line in inverted_summary:
            if re.search("^Data Sampling Rate:.*$", line):
                return int(line.split(" ")[-2])
            
    def extract_start_time(self):
        return self.parent_chb.full_summary[self.summary_start_line+1].split(" ")[-1]
    
    def extract_end_time(self):
        return self.parent_chb.full_summary[self.summary_start_line+2].split(" ")[-1]
    
    def extract_seizures(self):
        self.number_of_seizures = int(self.parent_chb.full_summary[self.summary_start_line+3].split(" ")[-1])
        if not self.number_of_seizures > 0: return {}
        seizure_def_block_start = self.summary_start_line + 4
        seizure_definition_lines = self.parent_chb.full_summary[ seizure_def_block_start 
                                                          : seizure_def_block_start + (2 * self.number_of_seizures)]
        all_seizures = {}
        seizure_counter = 0
        for i in range(0, len(seizure_definition_lines), 2):
            seizure_counter += 1
            start = seizure_definition_lines[i].split(" ")[-2]
            end = seizure_definition_lines[i+1].split(" ")[-2]
            all_seizures.update({seizure_counter : {"start" : int(start), "end" : int(end)}})
        return all_seizures
   
    def extract_preictal_period(self):
        preictal = {}
        MINUITE = 60
        for seizure, times in self.seizure_dict.items():
            preictal.update({seizure : {"start" : int(max(0, times["start"] - 15 * MINUITE)), "end" : int(times["start"] - 1)}})
        return preictal 