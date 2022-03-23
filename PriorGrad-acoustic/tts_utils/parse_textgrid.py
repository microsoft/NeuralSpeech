# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import argparse
import json
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input path of textgrid")
    parser.add_argument("--output", help="output path of json")
    return parser.parse_args()


def remove_empty_lines(text):
    """remove empty lines"""
    assert (len(text) > 0)
    assert (isinstance(text, list))
    text = [t.strip() for t in text]
    if "" in text:
        text.remove("")
    return text


class TextGrid(object):
    def __init__(self, text):
        text = remove_empty_lines(text)
        self.text = text
        self.line_count = 0
        self._get_type()
        self._get_time_intval()
        self._get_size()
        self.tier_list = []
        self._get_item_list()

    def _extract_pattern(self, pattern, inc):
        """
        Parameters
        ----------
        pattern : regex to extract pattern
        inc : increment of line count after extraction
        Returns
        -------
        group : extracted info
        """
        try:
            group = re.match(pattern, self.text[self.line_count]).group(1)
            self.line_count += inc
        except AttributeError:
            raise ValueError("File format error at line %d:%s" % (self.line_count, self.text[self.line_count]))
        return group

    def _get_type(self):
        self.file_type = self._extract_pattern(r"File type = \"(.*)\"", 2)

    def _get_time_intval(self):
        self.xmin = self._extract_pattern(r"xmin = (.*)", 1)
        self.xmax = self._extract_pattern(r"xmax = (.*)", 2)

    def _get_size(self):
        self.size = int(self._extract_pattern(r"size = (.*)", 2))

    def _get_item_list(self):
        """Only supports IntervalTier currently"""
        for itemIdx in range(1, self.size + 1):
            tier = OrderedDict()
            item_list = []
            tier_idx = self._extract_pattern(r"item \[(.*)\]:", 1)
            tier_class = self._extract_pattern(r"class = \"(.*)\"", 1)
            if tier_class != "IntervalTier":
                raise NotImplementedError("Only IntervalTier class is supported currently")
            tier_name = self._extract_pattern(r"name = \"(.*)\"", 1)
            tier_xmin = self._extract_pattern(r"xmin = (.*)", 1)
            tier_xmax = self._extract_pattern(r"xmax = (.*)", 1)
            tier_size = self._extract_pattern(r"intervals: size = (.*)", 1)
            for i in range(int(tier_size)):
                item = OrderedDict()
                item["idx"] = self._extract_pattern(r"intervals \[(.*)\]", 1)
                item["xmin"] = self._extract_pattern(r"xmin = (.*)", 1)
                item["xmax"] = self._extract_pattern(r"xmax = (.*)", 1)
                item["text"] = self._extract_pattern(r"text = \"(.*)\"", 1)
                item_list.append(item)
            tier["idx"] = tier_idx
            tier["class"] = tier_class
            tier["name"] = tier_name
            tier["xmin"] = tier_xmin
            tier["xmax"] = tier_xmax
            tier["size"] = tier_size
            tier["items"] = item_list
            self.tier_list.append(tier)

    def toJson(self):
        _json = OrderedDict()
        _json["file_type"] = self.file_type
        _json["xmin"] = self.xmin
        _json["xmax"] = self.xmax
        _json["size"] = self.size
        _json["tiers"] = self.tier_list
        return json.dumps(_json, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = parse_args()
    input = args.input
    output = args.output
    text = []
    with open(input, "r") as f:
        text = f.readlines()
    if len(text) == 0:
        raise IOError("input textgrid file can't be empty")
    text = remove_empty_lines(text)
    textgrid = TextGrid(text)
    with open(output, "w") as f:
        f.write(textgrid.toJson())
