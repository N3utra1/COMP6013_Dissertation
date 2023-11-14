import os 
import re

from .EDF_File import EDF_File
verbose = False

class CHB:
    def __init__(self, dir_name=None, chb_mit_path=None, verbosity=False):
        global verbose
        verbose = verbosity
        self.name = dir_name
        self.path = os.path.join(chb_mit_path, self.name)
        self.summary_path = os.path.join(self.path, f"{self.name}-summary.txt")
        self.full_summary = self.load_summary()
        self.seizures = {
            #"filename"  : { 
            #       1 : { "start" : 2906, "end" : 3060},
            #       2 : { "start" : 4053, "end" : 4101},
            #}
        }
        self.edf_files = self.generate_edf_files()
        
    def __repr__(self):
        return f"""
        name: {self.name} 
        path: {self.path} 
        summary path: {self.summary_path} 
        seizures: {self.seizures}
        """
       
    def generate_edf_files(self):
        loaded_files = []
        for file in os.listdir(self.path):
            if not os.path.splitext(file)[1] == ".edf": continue
            edf_file = EDF_File(self, file) 
            loaded_files.append(edf_file)
            if verbose: print(repr(edf_file))
            
        #update self.seizures
        for edf in loaded_files:
            if not edf.seizure_dict == {}:
                self.seizures.update({edf.name : edf.seizure_dict})
            
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
            if re.search("^file name: .*$", line):
                edf = find_edf_file(line.split(" ")[-1])
                edf.start_time = self.full_summary[i+1]
                edf.end_time = self.full_summary[i+2]
                i += 2
            i += 1