import re

import pandas as pd

class FixedWidthVariables(object):

    def __init__(self, variables, index_base=0):
        self.variables = variables

        self.colspecs = variables[["start", "end"]] - index_base

        self.colspecs = self.colspecs.astype(int).values.tolist()
        self.names = variables["name"]

    def ReadFixedWidth(self, filename, **options):
        df = pd.read_fwf(
            filename, colspecs=self.colspecs, names=self.names, **options
        )
        return df

def ReadStataDct(dct_file, **options):
    type_map = dict(
        byte=int, int=int, long=int, float=float, double=float, numeric=float
    )

    var_info = []
    with open(dct_file, **options) as f:
        for line in f:
            match = re.search(r"_column\(([^)]*)\)", line)
            if not match:
                continue
            start = int(match.group(1))
            t = line.split()
            vtype, name, fstring = t[1:4]
            name = name.lower()
            if vtype.startswith("str"):
                vtype = str
            else:
                vtype = type_map[vtype]
            long_desc = " ".join(t[4:]).strip('"')
            var_info.append((start, vtype, name, fstring, long_desc))

    columns = ["start", "type", "name", "fstring", "desc"]
    variables = pd.DataFrame(var_info, columns=columns)

    variables["end"] = variables.start.shift(-1)
    variables.loc[len(variables, index_base=1)]

    dct = FixedWidthVariables(variables, index_base=1)

    return dct