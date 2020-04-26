import os
import json
from glob import glob
from tqdm import tqdm

class GetResults:
    _dict = {
        "int_id": [],
        "defs": {
            "hash": [],
            "batch": [],
            "act": [],
            "comp": [],
            "error": [],
            "succeed": [],
        },
        "test": {"fname": [], "fpath": []},
        "train": {"fname": [], "fpath": []},
    }

    def get_logfile(self, rpath="run*"):
        int_id = 0
        for path in glob(rpath):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".txt"):
                        self._dict["int_id"].append(int_id)  # Add an int id
                        int_id += 1
                        # Add the haslib-key of the current object
                        self._dict["defs"]["hash"].append(file[:-4])
                        # Split the path for getting:
                        folders = root.split("/")
                        # Batch series
                        self._dict["defs"]["batch"].append(folders[-3])
                        # Activation function
                        self._dict["defs"]["act"].append(folders[-2])
                        # Ration of test- vs. training-set
                        self._dict["defs"]["comp"].append(folders[-1])
                    elif file.endswith("train.log"):
                        self._dict["train"]["fname"].append(file)
                        self._dict["train"]["fpath"].append(os.path.join(root, file))
                    elif file.endswith("test.log"):
                        self._dict["test"]["fname"].append(file)
                        self._dict["test"]["fpath"].append(os.path.join(root, file))

    def save_json(self, fname="data_results.json"):
        with open(fname, "w+") as fp:
            json.dump(self._dict, fp, indent=4)

    def return_dict(self):
        return self._dict


class AnalyzeResult(GetResults):
    def __init__(self, fname="data_results.json", rpath="run*"):
        super(AnalyzeResult, self).__init__()

        self._dict = self.initialse(fname=fname, rpath=rpath)

    def initialse(self, fname, rpath):
        self.get_logfile(rpath=rpath)
        self.save_json(fname=fname)
        return self.return_dict()

    def analyze(self):
        for run_i in tqdm(self._dict["int_id"], total=len(self._dict["int_id"])):
            _file_1 = self._dict["test"]["fpath"][run_i]
            _file_2 = self._dict["train"]["fpath"][run_i]
    def export2csv(self):
        pass


if __name__ == "__main__":
    AnalyzeResult().analyze()
