import os
import json
from glob import glob


class GetResults:
    _dict = {
        "defs": {"hash": [], "batch": [], "act": [], "comp": []},
        "test": {"fname": [], "fpath": []},
        "train": {"fname": [], "fpath": []},
    }

    def get_logfile(self, rpath="run*"):
        for path in glob(rpath):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".txt"):
                        self._dict["defs"]["hash"].append(file[:-4])
                        folders = root.split("/")
                        self._dict["defs"]["batch"].append(folders[-3])
                        self._dict["defs"]["act"].append(folders[-2])
                        self._dict["defs"]["comp"].append(folders[-1])
                    elif file.endswith("train.log"):
                        self._dict["train"]["fname"].append(file)
                        self._dict["train"]["fpath"].append(os.path.join(root, file))
                    elif file.endswith("test.log"):
                        self._dict["test"]["fname"].append(file)
                        self._dict["test"]["fpath"].append(os.path.join(root, file))

    def save_json(self, fname="data_results.json"):
        with open(fname, "w+") as fp:
            json.dump(self._dict, fp)

class AnalyzeResult(GetResults):
    def __init__(self):
        super(AnalyzeResult, self).__init__()

    def initialse(self):
        self.get_logfile()
        self.save_json()
if __name__ == "__main__":
    #results = GetResults()
    #results.get_logfile()
    #results.save_json()
    AnalyzeResult().initialse()
