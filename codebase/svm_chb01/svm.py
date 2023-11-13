import mne
import os 
from pprint import pprint
import re

class CHB:
    def __init__(self, dir_name):
        self.name = dir_name
        self.path = os.path.join(chb_mit_path, self.name)
        self.summary_path = os.path.join(self.path, f"{self.name}-summary.txt")
        self.seizures = {
            #"filename"  : [
            #    { "start" : 2906, "end" : 3060},
            #    { "start" : 4053, "end" : 4101},
            #]
            
            # "EDF_File.name" : "EDF_File.seizure_dict()"
        }
        self.full_summary = (self.load_summary()) 
        self.edf_files = self.generate_edf_files()
        
    def __repr__(self):
        return f"""
        Name: {self.name} 
        Path: {self.path} 
        Summary Path: {self.summary_path} 
        Seizures: {self.seizures}
        """
       
    def generate_edf_files(self):
        loaded_files = []
        for file in os.listdir(self.path):
            if not os.path.splitext(file)[1] == ".edf": continue
            edf_file = EDF_File(self, file) 
            print("loaded edf file:")
            print(repr(edf_file))
        return loaded_files  
        
    def load_summary(self):
        lines = []
        with open(self.summary_path, "r") as summary:
            lines = summary.readlines()
        lines = [line.strip() for line in lines if not line == "\n"]
        return lines
    
    def extract_data_sampling_rate(self):
        return self.full_summary[0].split(" ")[-2]
    
    
    def extract_edf_file_info(self):
        def find_edf_file(target_name):
            for edf in self.edf_files:
                if edf.name == target_name:
                    return  edf
        
        i = 0 
        for i in range(0, len(self.full_summary)):
            line = self.full_summary[i]
            if re.search("^File Name: .*$", line):
                edf = find_edf_file(line.split(" ")[-1])
                edf.start_time = self.full_summary[i+1]
                edf.end_time = self.full_summary[i+2]
                i += 2
            i += 1
                
class EDF_File():
    def __init__(self, _parent_chb, _edf_path):
        self.parent_chb = _parent_chb
        self.full_name = _edf_path
        self.name = _edf_path[6:]
        self.path = os.path.join(self.parent_chb.path, _edf_path)
        self.summary_start_line = self.get_summary_start_line() 
        self.channels = self.extract_channels()
        self.data_sampling_rate = self.extract_data_sampling_rate()
        self.start_time = self.extract_start_time()
        self.end_time = self.extract_end_time() 
        self.number_of_seizures = 0 #updated in self.extract_seizures()
        self.seizure_dict = self.extract_seizures()
        
    
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
        """

    def get_summary_start_line(self):
        for line in self.parent_chb.full_summary:
            if re.search(f"^File Name: {self.name}.*$", line): 
                return self.parent_chb.full_summary.index(line)
            
    def extract_channels(self):
        summary = self.parent_chb.full_summary[0: self.summary_start_line] 
        i = len(summary)-1
        start_index = None
        end_index = None
        while i  > 0:
            line = summary[i]
            if re.search("^Channel .*$", line):
                if end_index == None: end_index = i
            else:
                if not end_index == None:
                    return [line.split(" ")[-1] for line in summary[i + 1: end_index]]
            i -= 1 
    
    def extract_data_sampling_rate(self):
        summary = self.parent_chb.full_summary[0: self.summary_start_line]
        inverted_summary = summary[::-1] 
        for line in inverted_summary:
            if re.search("^Data Sampling Rate:.*$", line):
                return line.split(" ")[-2]
            
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
        for i in range(0, len(seizure_definition_lines), 2):
            start_line = seizure_definition_lines[i]
            end_line = seizure_definition_lines[i+1]
            print(start_line)
            print(end_line)
        

chb_mit_path = "../../chb-mit-scalp-eeg-database-1.0.0"
loaded_chb = []
for dir in os.listdir(chb_mit_path):
    chb = CHB(dir) 
    loaded_chb.append(chb)
    
for chb in loaded_chb:
    print(repr(chb))




# raw = mne.io.read_raw_edf(path)

# print(raw)
# print(raw.info)
# # raw.plot()
# # input()

# sname = get_file_name(path)
# print(path)
# print(sname)