#  LLEPE: Liquid-Liquid Equilibrium Parameter Estimator
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See LICENSE for more details.

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax:
        ax = plt.gca()
    left = ax.figure.subplotpars.left
    right = ax.figure.subplotpars.right
    top = ax.figure.subplotpars.top
    bottom = ax.figure.subplotpars.bottom
    fig_width = float(w) / (right - left)
    fig_height = float(h) / (top - bottom)
    ax.figure.set_size_inches(fig_width, fig_height)


def get_xml_value(info_dict, xml_filename):
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    d = info_dict
    value = None
    if (d['upper_attrib_name'] is not None
            and d['lower_attrib_name'] is not None):
        for child1 in root.iter(d['upper_element_name']):
            if (child1.attrib[d['upper_attrib_name']]
                    == d['upper_attrib_value']):
                for child2 in child1.iter(d['lower_element_name']):
                    if (child1.attrib[d['lower_attrib_name']]
                            == d['lower_attrib_value']):
                        value = child2.text

    elif d['upper_attrib_name'] is None and d['lower_attrib_name'] is not None:
        for child1 in root.iter(d['upper_element_name']):
            for child2 in child1.iter(d['lower_element_name']):
                if (child1.attrib[d['lower_attrib_name']]
                        == d['lower_attrib_value']):
                    value = child2.text
    elif d['upper_attrib_name'] is not None and d['lower_attrib_name'] is None:
        for child1 in root.iter(d['upper_element_name']):
            if (child1.attrib[d['upper_attrib_name']]
                    == d['upper_attrib_value']):
                for child2 in child1.iter(d['lower_element_name']):
                    value = child2.text
    else:
        for child1 in root.iter(d['upper_element_name']):
            for child2 in child1.iter(d['lower_element_name']):
                value = child2.text
    return value
